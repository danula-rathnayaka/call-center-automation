import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.graph.ingestion_workflow import ingestion_app

output_dir = os.path.join(os.path.dirname(__file__), "docs")
os.makedirs(output_dir, exist_ok=True)

rag_png = rag_app.get_graph().draw_mermaid_png()
rag_path = os.path.join(output_dir, "rag_workflow_graph.png")
with open(rag_path, "wb") as f:
    f.write(rag_png)
print(f"RAG workflow graph saved: {rag_path}")

ingestion_png = ingestion_app.get_graph().draw_mermaid_png()
ingestion_path = os.path.join(output_dir, "ingestion_workflow_graph.png")
with open(ingestion_path, "wb") as f:
    f.write(ingestion_png)
print(f"Ingestion workflow graph saved: {ingestion_path}")
