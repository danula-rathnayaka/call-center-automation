import time
import uuid

from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.embeddings import EmbeddingManager
from multiagent_rag.utils.sparse import SparseEmbeddingManager
from multiagent_rag.utils.logger import get_logger

logger = get_logger("retriever_main")


def initialise_system():
    logger.info("Initializing support system components")

    start_time = time.time()

    logger.info("Establishing connection to Pinecone database")
    _ = PineconeClient()

    logger.info("Loading dense embedding model (All-MiniLM)")
    _ = EmbeddingManager()

    logger.info("Initializing sparse encoder parameters")
    _ = SparseEmbeddingManager()

    elapsed = time.time() - start_time
    logger.info(f"System initialization complete in {elapsed:.2f} seconds")


def retriever():
    initialise_system()
    print("\n--- TELECOM AI SUPPORT SYSTEM ---")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Session ID: {thread_id}")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("-" * 50)
        print("AI Assistant: ", end="", flush=True)

        for event in rag_app.stream(
                {"query": user_query},
                config=config,
                stream_mode="updates"
        ):
            for node_name, output in event.items():
                if "final_answer" in output:
                    print(output["final_answer"], end="", flush=True)

        print("\n" + "-" * 50)


if __name__ == '__main__':
    retriever()
