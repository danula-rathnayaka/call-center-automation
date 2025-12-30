import os

from multiagent_rag.graph.ingestion_workflow import ingestion_app

if __name__ == "__main__":
    pdf_path = os.path.join("data", "LankaLink_Customer_Support_Manual.pdf")

    if not os.path.exists(pdf_path):
        exit()

    result = ingestion_app.invoke({
        "file_path": pdf_path,
        "chunks": [],
        "status": "start"
    })
