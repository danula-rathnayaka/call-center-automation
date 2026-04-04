import os

from fastapi import APIRouter, HTTPException

from api.schemas import (
    KnowledgeBaseStatus,
    KnowledgeResetResponse,
    IngestedDocumentsResponse,
    IngestedURLsResponse,
    DocumentEntry,
    DeleteKnowledgeResponse,
)
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_upload_dir = os.path.join(_project_root, "data", "uploads")


@router.get(
    "/status",
    response_model=KnowledgeBaseStatus,
    summary="Get knowledge base index statistics",
    description=(
        "Returns the name of the Pinecone index and the total number of vector chunks currently stored. "
        "Use this as an at-a-glance indicator of whether the knowledge base is populated. "
        "A `total_vectors` of 0 means no documents have been ingested yet and the AI will have no context to retrieve from."
    ),
)
async def get_knowledge_status():
    try:
        client = PineconeClient()
        index = client._index
        index_name = client._index_name

        stats = index.describe_index_stats()

        return KnowledgeBaseStatus(
            index_name=index_name,
            total_vectors=stats.get("total_vector_count", 0),
            status="ready",
        )

    except Exception as e:
        logger.error(f"Knowledge status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge base status: {str(e)}")


@router.delete(
    "/reset",
    response_model=KnowledgeResetResponse,
    summary="Wipe the entire knowledge base",
    description=(
        "**Destructive operation.** Deletes every vector from the Pinecone index. "
        "All ingested documents and URLs will need to be re-ingested after this action. "
        "BM25 sparse encoder will also lose its corpus. "
        "Intended for admin use only — show a confirmation dialog before calling this from the UI."
    ),
)
async def reset_knowledge_base():
    try:
        client = PineconeClient()
        client.delete_all()

        return KnowledgeResetResponse(
            status="completed",
            message="Knowledge base has been successfully wiped. All vectors deleted."
        )

    except Exception as e:
        logger.error(f"Knowledge base reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset knowledge base: {str(e)}")


@router.get(
    "/documents",
    response_model=IngestedDocumentsResponse,
    summary="List all ingested PDF and DOCX documents",
    description=(
        "Returns a deduplicated list of every PDF and DOCX file whose content has been ingested into "
        "the knowledge base. Each entry includes the source filename, document type, and document hash. "
        "Use this to render a \"Manage Documents\" list in the admin panel so users can see what files "
        "are currently powering the AI and delete individual ones if needed."
    ),
)
async def list_ingested_documents():
    """Return every unique document (PDF / DOCX) ingested into the knowledge base."""
    try:
        client = PineconeClient()
        raw: list = []
        for doc_type in ("pdf", "docx"):
            raw.extend(client.list_by_type(doc_type))

        entries = [
            DocumentEntry(
                source=r["source"],
                type=r["type"],
                document_hash=r["document_hash"],
            )
            for r in raw
        ]

        logger.info(f"Returning {len(entries)} ingested documents")
        return IngestedDocumentsResponse(total=len(entries), documents=entries)

    except Exception as e:
        logger.error(f"Failed to list ingested documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get(
    "/urls",
    response_model=IngestedURLsResponse,
    summary="List all ingested web-scraped URLs",
    description=(
        "Returns a deduplicated list of every seed URL that has been crawled and ingested into the knowledge base. "
        "Each entry represents a unique crawled domain/page — individual sub-pages scraped from the same seed "
        "are collapsed into one entry. Use this in the admin panel alongside the documents list "
        "to give a complete view of all knowledge sources."
    ),
)
async def list_ingested_urls():
    """Return every unique URL (web-scraped page) ingested into the knowledge base."""
    try:
        client = PineconeClient()
        raw = client.list_by_type("web")

        entries = [
            DocumentEntry(
                source=r["source"],
                type=r["type"],
                document_hash=r["document_hash"],
            )
            for r in raw
        ]

        logger.info(f"Returning {len(entries)} ingested URLs")
        return IngestedURLsResponse(total=len(entries), urls=entries)

    except Exception as e:
        logger.error(f"Failed to list ingested URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list URLs: {str(e)}")


@router.delete(
    "/documents/{source}",
    response_model=DeleteKnowledgeResponse,
    summary="Delete a document from the knowledge base by filename",
    description=(
        "Removes all Pinecone vectors whose `source` metadata matches the given filename (e.g. `MyDoc.pdf`). "
        "Also deletes the physical uploaded file from disk if it exists. "
        "Pass the exact filename returned from `GET /api/knowledge/documents` as the path parameter. "
        "Returns the count of vectors deleted. Returns `404` if no vectors match that source name."
    ),
)
async def delete_document(source: str):
    """Delete a document (PDF/DOCX) from the knowledge base by its source filename.
    Also removes the physical file from the uploads directory."""
    try:
        client = PineconeClient()
        deleted = client.delete_by_source(source)

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No vectors found for document source='{source}'. Nothing deleted."
            )

        physical_path = os.path.join(_upload_dir, source)
        file_removed = False
        if os.path.isfile(physical_path):
            os.remove(physical_path)
            file_removed = True
            logger.info(f"Removed uploaded file: {physical_path}")

        msg = f"Deleted {deleted} vectors for '{source}'."
        if file_removed:
            msg += " Physical file removed from uploads."

        return DeleteKnowledgeResponse(
            status="deleted",
            source=source,
            vectors_deleted=deleted,
            message=msg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document '{source}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete(
    "/urls",
    response_model=DeleteKnowledgeResponse,
    summary="Delete a scraped URL and all its pages from the knowledge base",
    description=(
        "Removes all Pinecone vectors whose `source` metadata matches the given URL. "
        "Pass the full seed URL (e.g. `https://example.com`) as the `source` **query parameter**. "
        "Note: use a query parameter, not a path segment, because URLs contain slashes. "
        "Returns the count of vectors deleted. Returns `404` if no vectors match that URL."
    ),
)
async def delete_url(source: str):
    """Delete all vectors for a scraped URL from the knowledge base.
    Pass the full seed URL as the `source` query parameter."""
    try:
        if not source.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="source must be a valid http/https URL")

        client = PineconeClient()
        deleted = client.delete_by_source(source)

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No vectors found for URL source='{source}'. Nothing deleted."
            )

        return DeleteKnowledgeResponse(
            status="deleted",
            source=source,
            vectors_deleted=deleted,
            message=f"Deleted {deleted} vectors for URL '{source}'.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete URL '{source}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete URL: {str(e)}")
