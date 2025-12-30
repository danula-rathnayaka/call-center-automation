from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from multiagent_rag.agents.pdf_ingestor import PDFIngestor
from multiagent_rag.state.ingestion_state import IngestionState
from multiagent_rag.utils.db_client import PineconeClient


def pdf_extraction_node(state: IngestionState):
    file_path = state["file_path"]

    agent = PDFIngestor()
    chunks = agent.process(file_path)

    if not chunks:
        return {"chunks": [], "status": "failed"}

    return {"chunks": chunks, "status": "extracted"}


def save_to_db_node(state: IngestionState):
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


workflow = StateGraph(IngestionState)

workflow.add_node("pdf_agent", pdf_extraction_node)
workflow.add_node("db_saver", save_to_db_node)

workflow.set_entry_point("pdf_agent")
workflow.add_edge("pdf_agent", "db_saver")
workflow.add_edge("db_saver", END)

ingestion_app = workflow.compile()
