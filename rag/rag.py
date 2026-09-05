"""
rag/rag.py
==========
Agentic RAG pipeline over PubMed / PMC (NCBI E-utilities, public + free API).

Pipeline:  query -> retrieval (PubMed esearch/esummary/efetch) -> reranking
           (cross-encoder-ish LLM relevance scoring) -> evidence assembly with
           metadata & citations -> LLM synthesis -> sufficiency check -> loop
           (agentic: retries with a reformulated query if evidence is weak).

Kept dependency-light: uses `httpx` for NCBI calls and the Anthropic API for
reranking/synthesis judgments (via a small helper), so it works standalone or
inside the LangGraph agent graph.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("medagent.rag")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # optional, raises rate limit from 3/s to 10/s
TRUSTED_DOMAINS = ["ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov", "nih.gov"]

MAX_RETRIEVAL_ROUNDS = 2  # agentic re-retrieval budget
CANDIDATES_PER_QUERY = 8
TOP_K_AFTER_RERANK = 5
MIN_SUFFICIENT_EVIDENCE = 2  # need at least this many reasonably-relevant docs


@dataclass
class Evidence:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str
    authors: list[str]
    url: str
    relevance_score: float = 0.0
    rerank_reason: str = ""

    def to_citation(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "authors": self.authors[:3] + (["et al."] if len(self.authors) > 3 else []),
            "url": self.url,
            "relevance_score": round(self.relevance_score, 3),
        }


@dataclass
class RAGResult:
    query: str
    reformulated_queries: list[str]
    evidence: list[Evidence]
    answer: str
    citations: list[dict[str, Any]]
    retrieval_rounds: int
    sufficient_evidence_found: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Retrieval — NCBI E-utilities (PubMed)
# --------------------------------------------------------------------------

async def _esearch(client: httpx.AsyncClient, query: str, retmax: int) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = await client.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


async def _efetch(client: httpx.AsyncClient, pmids: list[str]) -> list[Evidence]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = await client.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    return _parse_pubmed_xml(r.text)


def _parse_pubmed_xml(xml_text: str) -> list[Evidence]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID", default="")
            title = article.findtext(".//ArticleTitle", default="") or ""
            abstract_parts = [el.text or "" for el in article.findall(".//AbstractText")]
            abstract = " ".join(abstract_parts).strip() or "No abstract available."
            journal = article.findtext(".//Journal/Title", default="") or ""
            year = article.findtext(".//PubDate/Year", default="") or article.findtext(".//PubDate/MedlineDate", default="")
            authors = []
            for author in article.findall(".//Author")[:6]:
                last = author.findtext("LastName", default="")
                fore = author.findtext("ForeName", default="")
                if last:
                    authors.append(f"{fore} {last}".strip())
            out.append(Evidence(
                pmid=pmid, title=re.sub(r"\s+", " ", title).strip(),
                abstract=re.sub(r"\s+", " ", abstract).strip()[:2000],
                journal=journal, pub_date=str(year), authors=authors,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed parsing a PubMed article record: {e}")
    return out


async def retrieve_pubmed(query: str, k: int = CANDIDATES_PER_QUERY) -> list[Evidence]:
    async with httpx.AsyncClient() as client:
        try:
            pmids = await _esearch(client, query, k)
            evidence = await _efetch(client, pmids)
            logger.info(f"PubMed retrieval for '{query}' -> {len(evidence)} docs")
            return evidence
        except Exception as e:
            logger.error(f"PubMed retrieval failed for query='{query}': {e}")
            return []


# --------------------------------------------------------------------------
# Reranking — lightweight lexical + LLM relevance scoring
# --------------------------------------------------------------------------

def _lexical_overlap_score(query: str, text: str) -> float:
    """Cheap deterministic fallback reranker: token-overlap Jaccard-ish score."""
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOPWORDS
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / (len(q_tokens) ** 0.5)


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are", "with",
    "what", "how", "does", "do", "can", "which", "was", "were", "be", "as", "by", "that",
}


async def rerank_evidence(query: str, evidence: list[Evidence], llm_call=None) -> list[Evidence]:
    """Reranks evidence. Uses an LLM scorer if `llm_call` (async callable) is provided,
    otherwise falls back to deterministic lexical overlap so the module works standalone."""
    if not evidence:
        return []

    if llm_call is not None:
        try:
            doc_list = "\n".join(
                f"[{i}] Title: {e.title}\nAbstract: {e.abstract[:400]}"
                for i, e in enumerate(evidence)
            )
            prompt = (
                f"Query: {query}\n\nCandidate documents:\n{doc_list}\n\n"
                "Score each document's relevance to the query from 0.0 (irrelevant) to 1.0 (highly relevant). "
                "Respond ONLY as JSON: a list of objects like "
                '[{"index": 0, "score": 0.8, "reason": "short reason"}]. No other text.'
            )
            raw = await llm_call(prompt)
            import json
            cleaned = raw.strip().strip("```json").strip("```").strip()
            scores = json.loads(cleaned)
            for s in scores:
                i = s.get("index")
                if i is not None and 0 <= i < len(evidence):
                    evidence[i].relevance_score = float(s.get("score", 0.0))
                    evidence[i].rerank_reason = s.get("reason", "")
            evidence.sort(key=lambda e: e.relevance_score, reverse=True)
            return evidence[:TOP_K_AFTER_RERANK]
        except Exception as e:
            logger.warning(f"LLM reranking failed ({e}), falling back to lexical reranking.")

    for e in evidence:
        e.relevance_score = round(_lexical_overlap_score(query, e.title + " " + e.abstract), 4)
        e.rerank_reason = "lexical overlap (fallback reranker)"
    evidence.sort(key=lambda e: e.relevance_score, reverse=True)
    return evidence[:TOP_K_AFTER_RERANK]


# --------------------------------------------------------------------------
# Agentic RAG orchestration
# --------------------------------------------------------------------------

def _sufficiency_check(evidence: list[Evidence]) -> bool:
    strong = [e for e in evidence if e.relevance_score >= 0.35]
    return len(strong) >= MIN_SUFFICIENT_EVIDENCE


async def _reformulate_query(original: str, llm_call) -> str:
    if llm_call is None:
        # Deterministic fallback: broaden by stripping the most specific clause
        return re.sub(r"\b(in|for|with)\b.*$", "", original).strip() or original
    try:
        prompt = (
            f"The following medical search query returned insufficient relevant PubMed evidence: "
            f"'{original}'. Rewrite it as a broader or differently-phrased PubMed search query "
            "(use MeSH-friendly terms if helpful). Respond with ONLY the new query text, nothing else."
        )
        new_q = await llm_call(prompt)
        return new_q.strip().strip('"')
    except Exception:
        return original


async def _synthesize_answer(query: str, evidence: list[Evidence], llm_call) -> str:
    if not evidence:
        return (
            "I could not find sufficient peer-reviewed evidence on PubMed/PMC to answer this "
            "question confidently. Please consult a medical professional or refine the question."
        )
    context = "\n\n".join(
        f"[{i+1}] {e.title} ({e.journal}, {e.pub_date}) — PMID {e.pmid}\n{e.abstract[:600]}"
        for i, e in enumerate(evidence)
    )
    if llm_call is None:
        # Deterministic extractive fallback if no LLM is wired
        top = evidence[0]
        return (
            f"Based on the top matching study ({top.title}, PMID {top.pmid}): "
            f"{top.abstract[:500]}... [See citations for full evidence list.]"
        )
    prompt = (
        f"You are a medical research assistant. Answer the question using ONLY the evidence below. "
        f"Cite sources inline using [1], [2], etc. matching the numbering. If evidence is insufficient "
        f"or conflicting, say so explicitly. Do not invent facts not present in the evidence.\n\n"
        f"Question: {query}\n\nEvidence:\n{context}\n\nAnswer:"
    )
    try:
        return await llm_call(prompt)
    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")
        top = evidence[0]
        return f"(LLM synthesis unavailable — extractive fallback) {top.title}: {top.abstract[:400]}"


async def agentic_rag_query(query: str, llm_call=None, max_rounds: int = MAX_RETRIEVAL_ROUNDS) -> RAGResult:
    """
    Full agentic RAG loop:
      1. retrieve from PubMed
      2. rerank
      3. check sufficiency -> if weak, reformulate query and retry (bounded)
      4. synthesize grounded answer with citations
    `llm_call`: optional async callable(prompt:str) -> str, used for reranking/reformulation/
    synthesis. If None, deterministic fallbacks are used end-to-end (fully offline-testable).
    """
    reformulations = [query]
    all_evidence: list[Evidence] = []
    rounds = 0
    current_query = query

    for round_i in range(max_rounds):
        rounds += 1
        candidates = await retrieve_pubmed(current_query)
        reranked = await rerank_evidence(current_query, candidates, llm_call=llm_call)
        # merge, dedupe by pmid
        seen = {e.pmid for e in all_evidence}
        for e in reranked:
            if e.pmid not in seen:
                all_evidence.append(e)
                seen.add(e.pmid)

        if _sufficiency_check(all_evidence):
            break
        if round_i < max_rounds - 1:
            current_query = await _reformulate_query(current_query, llm_call)
            reformulations.append(current_query)
            logger.info(f"Insufficient evidence, reformulated query -> '{current_query}'")

    all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
    top_evidence = all_evidence[:TOP_K_AFTER_RERANK]
    sufficient = _sufficiency_check(top_evidence)

    answer = await _synthesize_answer(query, top_evidence, llm_call)

    return RAGResult(
        query=query,
        reformulated_queries=reformulations,
        evidence=top_evidence,
        answer=answer,
        citations=[e.to_citation() for e in top_evidence],
        retrieval_rounds=rounds,
        sufficient_evidence_found=sufficient,
        metadata={
            "total_candidates_considered": len(all_evidence),
            "trusted_source": "PubMed/PMC (NCBI E-utilities)",
        },
    )
