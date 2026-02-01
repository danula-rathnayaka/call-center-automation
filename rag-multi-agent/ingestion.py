import os

from multiagent_rag.graph.ingestion_workflow import ingestion_app


def ingestion():
    pdf_path = os.path.join("data", "SIM_Card_Service_Support_Manual.docx")

    if not os.path.exists(pdf_path):
        exit()

    result = ingestion_app.invoke({
        "file_path": pdf_path,
        "chunks": [],
        "status": "start"
    })

    print(result)


if __name__ == "__main__":
    ingestion()
