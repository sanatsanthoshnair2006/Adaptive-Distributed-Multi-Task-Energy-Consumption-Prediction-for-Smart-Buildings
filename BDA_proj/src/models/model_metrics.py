"""
Model Metrics Module
=====================

Defines the ``ModelMetrics`` dataclass that stores every measurement
produced during model evaluation, plus ``CandidateResult`` which augments
raw metrics with normalised values, composite scores, and ranking info.

Design Decisions:
    - Dataclasses over dicts:  Typed, IDE-friendly, and self-documenting.
      Converting to dict is trivial when Parquet persistence is needed.
    - Separation of raw vs. normalised:  Raw metrics are always preserved
      regardless of the normalisation strategy, making post-hoc analysis
      possible without re-running models.
    - Optional fields:  CPU / memory come from psutil which may be
      unavailable in some environments; ``Optional[float]`` makes this
      explicit.
    - ``FailedCandidateResult``:  A sentinel that records why a model
      failed without contaminating the ranking logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Raw performance metrics
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """
    Raw evaluation metrics for a single trained model on a single task.

    Attributes:
        task_name       : Prediction task identifier.
        model_name      : Model identifier (e.g. ``"random_forest"``).
        rmse            : Root Mean Squared Error.
        mae             : Mean Absolute Error.
        r2              : Coefficient of determination R².
        training_time   : Wall-clock training duration in seconds.
        prediction_time : Wall-clock prediction duration in seconds.
        memory_usage_mb : Peak RSS memory usage delta during training (MB).
                          ``None`` if psutil is unavailable.
        cpu_usage_pct   : Average CPU utilisation during training (percent).
                          ``None`` if psutil is unavailable.
        experiment_id   : UUID string of the enclosing experiment run.
        selection_profile : Active selection profile name.
    """

    task_name: str
    model_name: str
    rmse: float
    mae: float
    r2: float
    training_time: float
    prediction_time: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    experiment_id: str = ""
    selection_profile: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for serialisation."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"ModelMetrics(task={self.task_name!r}, model={self.model_name!r}, "
            f"rmse={self.rmse:.6f}, mae={self.mae:.6f}, r2={self.r2:.6f}, "
            f"train_t={self.training_time:.3f}s, pred_t={self.prediction_time:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Candidate result (raw + normalised + composite score + rank)
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """
    Full evaluation record for one candidate model.

    Extends ``ModelMetrics`` with normalised values, the composite
    performance score, and the final selection rank.

    Attributes:
        metrics              : Raw ``ModelMetrics`` object.
        normalised_rmse      : Min-max normalised RMSE in [0, 1].
        normalised_mae       : Min-max normalised MAE in [0, 1].
        normalised_r2_penalty: Normalised R² penalty (1 - normalised_R²).
        normalised_training_time  : Normalised training duration.
        normalised_prediction_time: Normalised prediction duration.
        normalised_memory_mb : Normalised memory usage.
        normalised_cpu_pct   : Normalised CPU usage.
        composite_score      : Weighted composite performance score.
                               *Lower is better.*
        rank                 : Integer rank (1 = best).
        is_selected          : True for the winning model.
        profile_weights      : Snapshot of the weight dict used.
    """

    metrics: ModelMetrics
    normalised_rmse: float = 0.0
    normalised_mae: float = 0.0
    normalised_r2_penalty: float = 0.0
    normalised_training_time: float = 0.0
    normalised_prediction_time: float = 0.0
    normalised_memory_mb: float = 0.0
    normalised_cpu_pct: float = 0.0
    composite_score: float = 0.0
    rank: int = 0
    is_selected: bool = False
    profile_weights: Dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten the nested structure into a single dict suitable for a
        DataFrame row or CSV/Parquet record.
        """
        raw = self.metrics.to_dict()
        raw.update({
            "normalised_rmse": self.normalised_rmse,
            "normalised_mae": self.normalised_mae,
            "normalised_r2_penalty": self.normalised_r2_penalty,
            "normalised_training_time": self.normalised_training_time,
            "normalised_prediction_time": self.normalised_prediction_time,
            "normalised_memory_mb": self.normalised_memory_mb,
            "normalised_cpu_pct": self.normalised_cpu_pct,
            "composite_score": self.composite_score,
            "rank": self.rank,
            "is_selected": self.is_selected,
            "profile_weights_json": json.dumps(self.profile_weights),
        })
        return raw

    def __repr__(self) -> str:
        return (
            f"CandidateResult(model={self.metrics.model_name!r}, "
            f"composite={self.composite_score:.6f}, rank={self.rank}, "
            f"selected={self.is_selected})"
        )


# ---------------------------------------------------------------------------
# Failed candidate (for error handling — does not participate in ranking)
# ---------------------------------------------------------------------------

@dataclass
class FailedCandidateResult:
    """
    Records the failure of a candidate model without breaking the pipeline.

    Attributes:
        task_name   : Prediction task identifier.
        model_name  : Model that failed.
        error       : Exception message.
        experiment_id: UUID of the enclosing experiment.
    """

    task_name: str
    model_name: str
    error: str
    experiment_id: str = ""

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to dict for persistence alongside successful candidates."""
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "failed": True,
            "error": self.error,
            "experiment_id": self.experiment_id,
        }
