"""API Router for Fast API."""
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from src.api.routes import hello, data

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, tags=["Dataset"])  # Include the dataset route


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")