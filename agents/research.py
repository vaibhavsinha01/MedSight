"""Compact PubMed retrieval + Gemini grounded synthesis."""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from typing import Any
import httpx
from dotenv import load_dotenv

load_dotenv()
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


async def retrieve_pubmed(query: str, limit: int = 4) -> list[dict[str, str]]:
    """Use PubMed relevance ranking to retrieve compact, citable abstracts."""
    params: dict[str, Any] = {"db": "pubmed", "term": query[:500], "retmax": limit, "sort": "relevance", "retmode": "json"}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "MedicalModelStudio/1.0 (research prototype)"}) as client:
        response = await client.get(f"{NCBI}/esearch.fcgi", params=params)
        response.raise_for_status()
        pmids = response.json().get("esearchresult", {}).get("idlist", [])[:limit]
        if not pmids:
            return []
        response = await client.get(f"{NCBI}/efetch.fcgi", params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        response.raise_for_status()
    root = ET.fromstring(response.text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract = " ".join(_text(n) for n in article.findall(".//Abstract/AbstractText"))[:1800]
        journal = _text(article.find(".//Journal/Title"))
        year = _text(article.find(".//PubDate/Year")) or _text(article.find(".//PubDate/MedlineDate"))[:4]
        if title and abstract:
            records.append({"title": title, "abstract": abstract, "journal": journal, "year": year,
                            "pmid": pmid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    return records


async def research_answer(query: str, limit: int = 4) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Enter a medical research question.")
    papers = await retrieve_pubmed(query, limit)
    if not papers:
        return {"query": query, "answer": "No matching PubMed abstracts were found. Try broader search terms.", "citations": []}
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"query": query, "answer": "Add GEMINI_API_KEY to .env to synthesize retrieved evidence.", "citations": papers}
    evidence = "\n\n".join(f"[{i}] {p['title']} ({p['year']}; PMID {p['pmid']})\n{p['abstract']}" for i, p in enumerate(papers, 1))
    prompt = ("Answer the question using only the retrieved abstracts. Cite each factual claim as [n]. "
              "Give a concise synthesis and one evidence limitation. Do not diagnose or recommend treatment.\n\n"
              f"Question: {query[:500]}\n\nEvidence:\n{evidence}")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(url, headers={"x-goog-api-key": key}, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
        })
        response.raise_for_status()
        body = response.json()
    answer = "".join(p.get("text", "") for p in body.get("candidates", [{}])[0].get("content", {}).get("parts", []))
    return {"query": query, "answer": answer or "The language model returned no text.", "citations": papers}


class ResearchAgent:
    """Explicit, user-invoked retrieval-augmented medical literature assistant."""

    async def run(self, query: str, limit: int = 4) -> dict[str, Any]:
        return await research_answer(query, limit)
