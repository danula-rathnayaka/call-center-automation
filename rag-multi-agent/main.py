import os

from multiagent_rag.graph.ingestion_workflow import ingestion_app
from multiagent_rag.graph.rag_workflow import rag_app


def test_ingestion():
    pdf_path = os.path.join("data", "SIM_Card_Service_Support_Manual.docx")

    if not os.path.exists(pdf_path):
        exit()

    result = ingestion_app.invoke({
        "file_path": pdf_path,
        "chunks": [],
        "status": "start"
    })

    print(result)


def test_retriever():
    print("\n--- TELECOM AI SUPPORT SYSTEM ---")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("-" * 50)

        result = rag_app.invoke({
            "query": user_query,
            "retrieved_docs": [],
            "final_answer": "",
            "status": "start"
        })

        print("\nAI Assistant:")
        print(result["final_answer"])
        print("-" * 50)


if __name__ == "__main__":
    test_retriever()
