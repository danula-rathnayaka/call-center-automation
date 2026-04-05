from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.logger import get_logger

logger = get_logger("db_reset")

client = PineconeClient()
client.delete_all()
logger.info("Database wipe completed successfully. Ready for fresh hybrid ingestion.")
