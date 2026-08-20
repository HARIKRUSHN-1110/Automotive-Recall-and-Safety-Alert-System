"""
main.py: The FastAPI application entry point.

This file only does three things:
1. Creates the FastAPI app with metadata (powers the Swagger UI)
2. Adds middleware (CORS so browsers can call the API)
3. Includes the router from routes.py

To run locally:
    uvicorn src.api.main:app --reload --port 8000

Then can visit:
    http://localhost:8000/docs       ← Swagger interactive docs
    http://localhost:8000/redoc     ← ReDoc alternative docs
    http://localhost:8000/openapi.json  ← Raw OpenAPI schema
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from .routes import router
# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# App creation with full metadata
# All these fields appear in the Swagger UI header at /docs--

app = FastAPI(
    title="Automotive Recall & Safety Alert API",
    description="""
## 🚗 Automotive Recall Risk Prediction API

An AI-powered system that predicts automotive recalls before official announcements
by analyzing NHTSA complaint patterns.

### What this API does
- **Predicts** recall probability for any vehicle (make/model/year)
- **Returns** raw NHTSA complaint records for a vehicle
- **Returns** official recall history for a vehicle
- **Shows** system-wide dataset statistics

### How predictions work
The ML model was trained on historical NHTSA complaints and matched recalls.
It learned which complaint patterns (keywords, components, frequency) tend to
precede official recall announcements by 6-12 months.

### Data source
All data is sourced from the **NHTSA public API** (api.nhtsa.gov).
Data is refreshed daily.

### Rate limits
100 requests per hour per IP address.
    """,
    version="1.0.0",
    contact={
        "name": "Automotive Recall System",
        "url": "https://github.com/HARIKRUSHN-1110/automotive-recall-and-safety-alert-system",
    },
    license_info={
        "name": "MIT License",
    },
    # Where Swagger UI and ReDoc are served
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
# Allows browsers (and Streamlit app) to call this API without being blocked by the browser's same-origin policy

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict this to relevant domain
    allow_credentials=True,
    allow_methods=["GET"],     # This API is read-only, so GET only
    allow_headers=["*"],
)

# Include the router
# All 4 routes from routes.py are now mounted under /api/v1
# example: router has /predict so full path becomes /api/v1/predict

app.include_router(router)

# Root endpoint — helpful landing page so /  isn't a 404

@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint. Redirects developers to the docs.
    include_in_schema=False means it won't appear in Swagger.
    """
    return JSONResponse({
        "message": "Automotive Recall & Safety Alert API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "predict": "/api/v1/predict?make=BMW&model=3 Series&year=2020",
            "complaints": "/api/v1/complaints?make=BMW&model=3 Series&year=2020",
            "recalls": "/api/v1/recalls?make=BMW&model=3 Series&year=2020",
            "stats": "/api/v1/stats",
        }
    })

# Health check — used by deployment platforms to verify the app is running

@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok", "version": "1.0.0"}

# Run directly (python src/api/main.py) — for testing
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)