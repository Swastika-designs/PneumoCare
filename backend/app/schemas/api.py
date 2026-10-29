"""Pydantic request/response models for the typed parts of the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    database: bool


class GradCamPayload(BaseModel):
    overlay: str = Field(description="PNG data URL of the heat-map blended over the X-ray")
    heatmap: str = Field(description="PNG data URL of the raw heat-map")
    target_class: str
    method: str = "Grad-CAM on ResNet layer4"


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_label: str
    probability_pneumonia: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    threshold: float
    model_name: str
    image: str
    gradcam: GradCamPayload | None = None
    warnings: list[str] = []
    disclaimer: str


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: str = "tfidf"
    top_k: int = Field(default=10, ge=1, le=50)


class WsdRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    abbreviation: str | None = None
    variant: str = "corpus_lesk"


class Scenario(BaseModel):
    name: str = Field(default="Scenario", max_length=60)
    demand_change_pct: float = Field(default=0, ge=-90, le=500)
    staff_fte: float = Field(default=1, gt=0, le=1000)
    minutes_per_encounter: float | None = Field(default=None, gt=0, le=1440)


class ScenarioRequest(BaseModel):
    scenarios: list[Scenario] = Field(min_length=1, max_length=6)
    start_year: int = 2016
    hours_per_fte_month: float = Field(default=160, gt=0, le=400)
