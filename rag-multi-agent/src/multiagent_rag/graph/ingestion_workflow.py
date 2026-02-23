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
    chunks = agent.process(file_path)

    if not chunks:
        return {"chunks": [], "status": "failed"}
    return {"chunks": chunks, "status": "extracted"}


def doc_extraction_node(state: IngestionState):
    file_path = state["file_path"]

    logger.info(f"Starting DOCX extraction for: {file_path}")
    agent = DocIngestor()
    chunks = agent.process(file_path)

    if not chunks:
        return {"chunks": [], "status": "failed"}
    return {"chunks": chunks, "status": "extracted"}


def url_extraction_node(state: IngestionState):
    url = state["file_path"]
    logger.info(f"Starting URL extraction for: {url}")

    agent = URLIngestor()
    chunks = agent.process(url)

    if not chunks:
        return {"chunks": [], "status": "failed"}
    return {"chunks": chunks, "status": "extracted"}


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


workflow = StateGraph(IngestionState)

workflow.add_node("pdf_agent", pdf_extraction_node)
workflow.add_node("doc_agent", doc_extraction_node)
workflow.add_node("url_agent", url_extraction_node)
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

workflow.add_edge("pdf_agent", "db_saver")
workflow.add_edge("doc_agent", "db_saver")
workflow.add_edge("url_agent", "db_saver")

workflow.add_edge("db_saver", END)

ingestion_app = workflow.compile()
