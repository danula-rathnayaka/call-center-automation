from multiagent_rag.utils.db_client import PineconeClient

client = PineconeClient()
client.delete_all()
print("Database Wiped. Ready for Hybrid Ingestion.")
