"""Pydantic request/response schemas for the EduSentinel API."""

from pydantic import BaseModel, Field
from typing import Optional


class ChildProfile(BaseModel):
    age: int = Field(..., ge=6, le=15)
    gender: str = Field(..., pattern="^[MF]$")
    grade_level: int = Field(..., ge=1, le=9)
    attendance_rate: float = Field(..., ge=0, le=1)
    math_score: float = Field(..., ge=0, le=100)
    literacy_score: float = Field(..., ge=0, le=100)
    household_income_usd_day: float = Field(..., gt=0)
    parent_edu_level: int = Field(..., ge=0, le=3)
    sibling_count: int = Field(..., ge=0, le=14)
    distance_to_school_km: float = Field(..., gt=0)
    disability: int = Field(0, ge=0, le=1)
    conflict_displaced: int = Field(0, ge=0, le=1)
    school_fee_burden: float = Field(..., ge=0, le=1)
    has_birth_certificate: int = Field(1, ge=0, le=1)
    meal_programme_access: int = Field(0, ge=0, le=1)
    poverty_rate_lga: float = Field(..., ge=0, le=1)
    conflict_score_lga: float = Field(..., ge=0, le=1)
    teacher_pupil_ratio: float = Field(..., gt=0)


class DropoutPredictionResponse(BaseModel):
    dropout_probability: float
    risk_level: str
    top_factors: list[dict]
    intervention_priority: str
    recommended_programmes: list[str]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    language: str = Field("en", pattern="^(en|ha|yo|ig|pcm)$")
    grade_level: int = Field(4, ge=1, le=9)
    subject: str = "Mathematics"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    language: str
    sources: list[str]
    session_id: str


class LGAFilter(BaseModel):
    states: Optional[list[str]] = None
    zones: Optional[list[str]] = None
    risk_tiers: Optional[list[str]] = None
    min_oos_rate: float = 0.0
    max_oos_rate: float = 1.0
    dominant_driver: Optional[str] = None


class RetrainResponse(BaseModel):
    status: str
    experiment: str
    metrics: dict
    run_id: Optional[str] = None
