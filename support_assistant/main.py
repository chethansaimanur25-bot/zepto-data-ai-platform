from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import run
from ingest import ingest, get_model
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest()       # build the ChromaDB index if it does not exist yet
    get_model()    # load the local embedding model once at startup
    yield


app = FastAPI(title="Zepto Support Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return run(req.query)