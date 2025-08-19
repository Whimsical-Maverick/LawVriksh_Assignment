from fastapi import FastAPI
from src.routes.analyze import router as analyze_router
from src.routes.recommend import router as recommend_router

app = FastAPI(title="Agentic Blog Support System")

# Routers (secured internally)
app.include_router(analyze_router, prefix="/api", tags=["analyze"])
app.include_router(recommend_router, prefix="/api", tags=["recommend"])
