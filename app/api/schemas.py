"""Stage 19.1 -- pydantic request/response contracts for the /predict endpoint.

Mirrors the pandera contracts in climate_health.data.schemas (Stage 2) field
for field, column for column (same ID pattern, same categorical choices, same
numeric ranges) -- expressed twice because pandera validates dataframes and
pydantic validates JSON request bodies, not because the validation rules
differ. A request that would fail TEST_SCHEMA / CLIMATE_FEATURES_SCHEMA
fails pydantic the same way, as a 422, before the handler ever runs.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    ID: str = Field(pattern=r"^ID_[0-9A-F]{8}$")
    zone: Literal["Rural", "Peri_urban"]
    gender: Literal["Male", "Female"]
    deathdate: date
    age: float = Field(ge=0, le=120)
    avg_temperature: float = Field(ge=-10, le=50)
    max_temperature: float = Field(ge=-10, le=55)
    min_temperature: float = Field(ge=-10, le=45)
    precipitation: float = Field(ge=0)
    latitude: float = Field(ge=-2.0, le=5.0)
    longitude: float = Field(ge=29.0, le=36.0)
    location: str

    rain_sum_7d: float = Field(ge=0)
    rain_sum_30d: float = Field(ge=0)
    rain_sum_90d: float = Field(ge=0)
    rain_days_30d: int = Field(ge=0, le=30)
    max_daily_rain_30d: float = Field(ge=0)
    tavg_7d: float = Field(ge=-10, le=45)
    tavg_30d: float = Field(ge=-10, le=45)
    tavg_90d: float = Field(ge=-10, le=45)
    tmax_30d: float = Field(ge=-10, le=55)
    tmin_30d: float = Field(ge=-10, le=45)
    hot_days_30d: int = Field(ge=0, le=30)
    temp_range_mean_30d: float = Field(ge=0)
    ndvi_30d: float = Field(ge=-1, le=1)
    ndvi_90d: float = Field(ge=-1, le=1)
    elevation: float = Field(ge=0, le=5200)
    slope: float = Field(ge=0, le=90)


class PredictResponse(BaseModel):
    ID: str
    TargetF1: int
    TargetRAUC: float
