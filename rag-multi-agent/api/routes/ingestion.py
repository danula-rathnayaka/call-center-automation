import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import IngestionResponse, URLIngestionRequest
from multiagent_rag.graph.ingestion_workflow import ingestion_app
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

# Resolve the data/uploads directory relative to the project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_upload_dir = os.path.join(_project_root, "data", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


@router.post("/file", response_model=IngestionResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX file for ingestion into the knowledge base.

    The file will be:
    1. Saved to the server
    2. Processed by the appropriate extraction agent (PDF/DOCX)
    3. Chunked into semantic segments
    4. Embedded and stored in the vector database
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Save uploaded file
        file_path = os.path.join(_upload_dir, filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File saved: {file_path} ({len(content)} bytes)")

        # Run the ingestion workflow
        result = ingestion_app.invoke({
            "file_path": file_path,
            "chunks": [],
            "status": "start"
        })

        status = result.get("status", "unknown")

        if status == "completed":
            return IngestionResponse(
                status="completed",
                message=f"Successfully ingested '{filename}' into the knowledge base.",
            )
        elif status == "failed":
            return IngestionResponse(
                status="failed",
                message=f"Failed to extract content from '{filename}'. The file may be empty or corrupted.",
            )
        else:
            return IngestionResponse(
                status=status,
                message=f"Ingestion completed with status: {status}",
            )

    except Exception as e:
        logger.error(f"File ingestion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


@router.post("/url", response_model=IngestionResponse)
async def ingest_url(request: URLIngestionRequest):
    """
    Ingest content from a URL into the knowledge base.

    The URL will be:
    1. Scraped for textual content
    2. Chunked into semantic segments
    3. Embedded and stored in the vector database
    """
    url = request.url.strip()

    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Must start with http:// or https://"
        )

    try:
        result = ingestion_app.invoke({
            "file_path": url,
            "chunks": [],
            "status": "start"
        })

        status = result.get("status", "unknown")

        if status == "completed":
            return IngestionResponse(
                status="completed",
                message=f"Successfully ingested content from '{url}'.",
            )
        elif status == "failed":
            return IngestionResponse(
                status="failed",
                message=f"Failed to extract content from '{url}'.",
            )
        else:
            return IngestionResponse(
                status=status,
                message=f"Ingestion completed with status: {status}",
            )

    except Exception as e:
        logger.error(f"URL ingestion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest URL: {str(e)}"
        )
