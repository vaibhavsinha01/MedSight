"""
main.py
=======
Entrypoint. Run with:

    python main.py
    # or
    uvicorn main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""

import logging
import os

import uvicorn

from api import app  # noqa: F401  (re-exported for `uvicorn main:app`)

os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "app.log")),
    ],
)

if __name__ == "__main__":
    pass
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=bool(os.environ.get("DEV_RELOAD")))
