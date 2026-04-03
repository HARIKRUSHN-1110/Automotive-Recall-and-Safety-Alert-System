"""
schemas.py: Pydantic validation models for FastAPI.

complaints columns:
    odi_number, make, model, model_year, component, summary,
    crash, fire, injuries, deaths, date_complained,
    date_of_incident, vehicle_speed, created_at

recalls columns:
    campaign_number, manufacturer, make, model, model_year,
    component, summary, consequence, remedy, notes,
    recall_date, park_it, created_at
"""

from pydantic import BaseModel, Field
from typing import Optional

# /predict
class PredictResponse(BaseModel):
    make: str = Field(..., example="BMW")
    model: str = Field(..., example="3 Series")
    year: int = Field(..., example=2020)
    risk_score: float = Field(..., ge=0, le=100, example=68.5,
        description="Recall risk score 0 (safe) to 100 (high risk)")
    risk_label: str = Field(..., example="medium",
        description="low | medium | high")
    confidence: float = Field(..., ge=0, le=1, example=0.82)
    complaint_count: int = Field(..., example=143)
    top_components: list[str] = Field(default=[],
        example=["ENGINE", "FUEL SYSTEM"])
    message: str = Field(..., example="Prediction based on 143 complaints.")

# /complaints
class ComplaintRecord(BaseModel):
    odi_number: Optional[str] = Field(None, example="11234567")
    make: str = Field(..., example="BMW")
    model: str = Field(..., example="3 Series")
    year: int = Field(..., example=2020)
    component: Optional[str] = Field(None, example="ENGINE AND ENGINE COOLING")
    summary: Optional[str] = Field(None, example="Vehicle stalled at highway speed.")
    crash: Optional[bool] = Field(None)
    fire: Optional[bool] = Field(None)
    injuries: Optional[int] = Field(None, example=0)
    deaths: Optional[int] = Field(None, example=0)
    date_complained: Optional[str] = Field(None, example="2023-04-15")
    date_of_incident: Optional[str] = Field(None, example="2023-03-10")
    vehicle_speed: Optional[str] = Field(None, example="65 MPH")

class ComplaintsResponse(BaseModel):
    make: str
    model: str
    year: int
    total_found: int = Field(..., example=143)
    returned: int = Field(..., example=10)
    complaints: list[ComplaintRecord]

# /recalls
class RecallRecord(BaseModel):
    campaign_number: Optional[str] = Field(None, example="23V123000")
    manufacturer: Optional[str] = Field(None, example="BMW OF NORTH AMERICA, LLC")
    make: str = Field(..., example="BMW")
    model: str = Field(..., example="3 Series")
    year: int = Field(..., example=2020)
    component: Optional[str] = Field(None, example="FUEL SYSTEM, GASOLINE")
    summary: Optional[str] = Field(None, example="Fuel may leak from the pump.")
    consequence: Optional[str] = Field(None, example="Fire risk.")
    remedy: Optional[str] = Field(None, example="Dealers will replace the fuel pump.")
    notes: Optional[str] = Field(None)
    recall_date: Optional[str] = Field(None, example="2023-06-01")
    park_it: Optional[bool] = Field(None,
        description="Whether NHTSA recommends parking the vehicle until repaired")

class RecallsResponse(BaseModel):
    make: str
    model: str
    year: int
    total_recalls: int = Field(..., example=3)
    recalls: list[RecallRecord]

# /stats
class StatsResponse(BaseModel):
    total_complaints: int = Field(..., example=52341)
    total_recalls: int = Field(..., example=9812)
    manufacturers_covered: int = Field(..., example=24)
    date_range_start: Optional[str] = Field(None, example="2000-01-01")
    date_range_end: Optional[str] = Field(None, example="2025-12-31")
    last_updated: str = Field(..., example="2026-03-13T10:00:00")
    api_version: str = Field(default="1.0.0")