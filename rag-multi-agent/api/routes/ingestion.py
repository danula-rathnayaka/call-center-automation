import asyncio
import os
from typing import List

from api.schemas import IngestionResponse, URLIngestionRequest, BatchIngestionResponse
from fastapi import APIRouter, File, HTTPException, UploadFile

from multiagent_rag.graph.ingestion_workflow import ingestion_app
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.scrape_review_store import (
    get_pending_queue, get_queue_item,
    approve_item, reject_item, get_stats,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_upload_dir = os.path.join(_project_root, "data", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _run_ingestion(file_path: str, label: str) -> IngestionResponse:
    result = ingestion_app.invoke({
        "file_path": file_path,
        "chunks": [],
        "document_hash": "",
        "status": "start",
    })
    status = result.get("status", "unknown")

    if status == "completed":
        chunks = result.get("chunks", [])
        return IngestionResponse(
            status="completed",
            message=f"Successfully ingested '{label}' into the knowledge base.",
            chunks_ingested=len(chunks),
        )
    elif status == "duplicate":
        return IngestionResponse(
            status="duplicate",
            message=f"'{label}' has already been ingested. Skipping to avoid duplication.",
            chunks_ingested=0,
        )
    elif status == "failed":
        return IngestionResponse(
            status="failed",
            message=f"Failed to extract content from '{label}'.",
            chunks_ingested=0,
        )
    else:
        return IngestionResponse(
            status=status,
            message=f"Ingestion completed with status: {status}",
            chunks_ingested=0,
        )


@router.post("/file", response_model=IngestionResponse)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        file_path = os.path.join(_upload_dir, filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File saved: {file_path} ({len(content)} bytes)")
        return _run_ingestion(file_path, filename)

    except Exception as e:
        logger.error(f"File ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")


@router.post("/batch", response_model=BatchIngestionResponse)
async def ingest_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    successful = 0
    failed = 0
    duplicates = 0

    saved_paths = []
    for file in files:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            results.append(IngestionResponse(
                status="failed",
                message=f"Unsupported file type '{ext}' for '{filename}'.",
                chunks_ingested=0,
            ))
            failed += 1
            continue

        try:
            file_path = os.path.join(_upload_dir, filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_paths.append((file_path, filename))
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {str(e)}")
            results.append(IngestionResponse(
                status="failed",
                message=f"Failed to save '{filename}': {str(e)}",
                chunks_ingested=0,
            ))
            failed += 1

    loop = asyncio.get_event_loop()
    ingestion_tasks = [
        loop.run_in_executor(None, _run_ingestion, path, label)
        for path, label in saved_paths
    ]
    ingestion_results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)

    for res in ingestion_results:
        if isinstance(res, Exception):
            results.append(IngestionResponse(
                status="failed",
                message=str(res),
                chunks_ingested=0,
            ))
            failed += 1
        else:
            results.append(res)
            if res.status == "completed":
                successful += 1
            elif res.status == "duplicate":
                duplicates += 1
            else:
                failed += 1

    return BatchIngestionResponse(
        results=results,
        total_files=len(files),
        successful=successful,
        failed=failed,
        duplicates=duplicates,
    )


@router.post("/url", response_model=IngestionResponse)
async def ingest_url(request: URLIngestionRequest):
    url = request.url.strip()

    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

    try:
        return _run_ingestion(url, url)
    except Exception as e:
        logger.error(f"URL ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")


@router.get("/scrape/review")
async def get_scrape_review_queue():
    try:
        return {"items": get_pending_queue(), "stats": get_stats()}
    except Exception as e:
        logger.error(f"Failed to fetch review queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scrape/approve/{item_id}", response_model=IngestionResponse)
async def approve_scraped_page(item_id: int):
    try:
        item = get_queue_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
        if item["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"Item {item_id} is already {item['status']}")

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_ingestion, item["url"], item["url"]
        )

        approve_item(item_id)

        return IngestionResponse(
            status=result.status,
            message=f"Page '{item['url']}' approved and ingested. {result.message}",
            chunks_ingested=result.chunks_ingested,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scrape/reject/{item_id}")
async def reject_scraped_page(item_id: int):
    try:
        item = get_queue_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
        if item["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"Item {item_id} is already {item['status']}")

        reject_item(item_id)
        return {"status": "rejected", "item_id": item_id, "url": item["url"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
