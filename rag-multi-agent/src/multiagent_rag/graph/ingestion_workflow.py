from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from multiagent_rag.agents.doc_ingestor import DocIngestor
from multiagent_rag.agents.pdf_ingestor import PDFIngestor
from multiagent_rag.agents.url_ingestor import URLIngestor
from multiagent_rag.state.ingestion_state import IngestionState
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def pdf_extraction_node(state: IngestionState):
    file_path = state["file_path"]
    logger.info(f"Starting PDF extraction for: {file_path}")

    agent = PDFIngestor()
    chunks, doc_hash = agent.process(file_path)

    if not chunks:
        return {"chunks": [], "document_hash": doc_hash, "status": "failed"}
    return {"chunks": chunks, "document_hash": doc_hash, "status": "extracted"}


def doc_extraction_node(state: IngestionState):
    file_path = state["file_path"]

    logger.info(f"Starting DOCX extraction for: {file_path}")
    agent = DocIngestor()
    chunks, doc_hash = agent.process(file_path)

    if not chunks:
        return {"chunks": [], "document_hash": doc_hash, "status": "failed"}
    return {"chunks": chunks, "document_hash": doc_hash, "status": "extracted"}


def url_extraction_node(state: IngestionState):
    url = state["file_path"]
    logger.info(f"Starting URL extraction for: {url}")

    agent = URLIngestor()
    chunks, doc_hash = agent.process(url)

    if not chunks:
        return {"chunks": [], "document_hash": doc_hash, "status": "failed"}
    return {"chunks": chunks, "document_hash": doc_hash, "status": "extracted"}


def duplicate_checker_node(state: IngestionState):
    doc_hash = state.get("document_hash", "")

    if not doc_hash:
        return {"status": "extracted"}

    if state.get("status") != "extracted":
        return {}

    client = PineconeClient()
    if client.check_duplicate(doc_hash):
        logger.info(f"Duplicate document detected, skipping insertion. Hash: {doc_hash}")
        return {"status": "duplicate"}

    return {}


def save_to_db_node(state: IngestionState):
    logger.info("Initializing database save operation")
    chunks_data = state["chunks"]

    if not chunks_data:
        return {"status": "skipped"}

    documents = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in chunks_data
    ]

    client = PineconeClient()
    success = client.insert_documents(documents)

    status = "completed" if success else "db_error"
    return {"status": status}


def route_file_type(state: IngestionState):
    file_path = state["file_path"].lower()

    if file_path.startswith(("http://", "https://")):
        return "url_agent"
    elif file_path.endswith(".pdf"):
        return "pdf_agent"
    elif file_path.endswith((".docx", ".doc")):
        return "doc_agent"
    else:
        logger.warning(f"Unsupported input type detected: {file_path}")
        return "end"


def should_save(state: IngestionState):
    if state.get("status") == "duplicate":
        return "end"
    return "db_saver"


workflow = StateGraph(IngestionState)

workflow.add_node("pdf_agent", pdf_extraction_node)
workflow.add_node("doc_agent", doc_extraction_node)
workflow.add_node("url_agent", url_extraction_node)
workflow.add_node("duplicate_checker", duplicate_checker_node)
workflow.add_node("db_saver", save_to_db_node)

workflow.set_conditional_entry_point(
    route_file_type,
    {
        "pdf_agent": "pdf_agent",
        "doc_agent": "doc_agent",
        "url_agent": "url_agent",
        "end": END
    }
)

workflow.add_edge("pdf_agent", "duplicate_checker")
workflow.add_edge("doc_agent", "duplicate_checker")
workflow.add_edge("url_agent", "duplicate_checker")

workflow.add_conditional_edges(
    "duplicate_checker",
    should_save,
    {
        "db_saver": "db_saver",
        "end": END,
    }
)

workflow.add_edge("db_saver", END)

ingestion_app = workflow.compile()
