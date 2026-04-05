import os

from multiagent_rag.graph.ingestion_workflow import ingestion_app
from multiagent_rag.utils.logger import get_logger

logger = get_logger("ingestion_main")


def ingestion():
    pdf_path = os.path.join("data", "SIM_Card_Service_Support_Manual.docx")

    if not os.path.exists(pdf_path):
        exit()

    logger.info(f"Starting ingestion process for file: {pdf_path}")
    result = ingestion_app.invoke({
        "file_path": pdf_path,
        "chunks": [],
        "status": "start"
    })

    logger.info(f"Ingestion process completed with result: {result.get('status', 'unknown')}")


if __name__ == "__main__":
    ingestion()
