"""
Selection Strategy Module
==========================

Implements the Composite Performance Score normalisation and ranking logic
that is the core of the adaptive model selection framework.

Methodology
-----------
Because the seven evaluation metrics have different units and scales,
they cannot be summed directly.  Two normalisation strategies are applied
before combining:

Cost metrics (lower is better)
    Normalised using **min-max scaling**::

        normalised = (value - min) / (max - min)

    Result is in [0, 1]; 0 means best, 1 means worst.

Benefit metrics (higher is better)  →  converted to a penalty
    Inverted min-max::

        penalty = (max - value) / (max - min)

    Result is in [0, 1]; 0 means best (highest R²), 1 means worst.

Edge case
    When all candidates produce the same value for a metric
    (``max == min``), every candidate is assigned ``0.0`` for that metric
    (they are equally good / equally bad — the metric provides no
    discriminating information).

Composite Performance Score
    A weighted sum of all normalised values::

        score = Σ (weight_i × normalised_i)

    **Lower score = better model.**

Tie-breaking
    If two scores are within ``tie_tolerance`` of each other, the following
    secondary criteria are applied in order:
    1. Lower raw RMSE.
    2. Lower raw MAE.
    3. Lower raw prediction time.
    4. Lexicographic model name (deterministic alphabetical ordering).

Design Decisions:
    - Pure-Python:  No Spark actions here — all metrics have already been
      collected to the driver as floats before normalisation runs.
    - Stateless:  ``SelectionStrategy`` holds no mutable state; each
      ``rank()`` call is independent.
    - Configuration-driven:  Weights and tie tolerance come from
      ``model_config.yaml``; no hardcoding in Python.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from src.logging.logger_factory import LoggerFactory
from src.models.model_metrics import CandidateResult, ModelMetrics


class SelectionStrategyError(Exception):
    """Raised when the selection strategy encounters an invalid state."""


class SelectionStrategy:
    """
    Normalises raw metrics, computes Composite Performance Scores, and
    produces a deterministic ranked list of candidate models.

    Args:
        profile_weights (Dict[str, float]):  Weight for each metric key.
            Keys: ``rmse``, ``mae``, ``r2_penalty``, ``training_time``,
            ``prediction_time``, ``memory_usage``, ``cpu_usage``.
        tie_tolerance (float):  Score delta below which two candidates are
            considered tied.  Defaults to ``1e-6``.
        profile_name (str):  Human-readable name of the active profile.
    """

    # Metric keys exactly as they appear in the weights configuration
    _WEIGHT_KEYS = (
        "rmse",
        "mae",
        "r2_penalty",
        "training_time",
        "prediction_time",
        "memory_usage",
        "cpu_usage",
    )

    def __init__(
        self,
        profile_weights: Dict[str, float],
        tie_tolerance: float = 1e-6,
        profile_name: str = "balanced",
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._weights = profile_weights
        self._tie_tolerance = tie_tolerance
        self._profile_name = profile_name
        self._validate_weights()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def rank(
        self,
        raw_metrics: List[ModelMetrics],
    ) -> List[CandidateResult]:
        """
        Normalise metrics, compute composite scores, and rank candidates.

        Args:
            raw_metrics (List[ModelMetrics]):  One entry per candidate model.
                Must contain at least one element.

        Returns:
            List[CandidateResult]:  Ranked candidates (rank 1 = best).
                Sorted ascending by composite score (lower = better).

        Raises:
            SelectionStrategyError:  If ``raw_metrics`` is empty.
        """
        if not raw_metrics:
            raise SelectionStrategyError(
                "Cannot rank: no raw metrics provided."
            )

        self._logger.info(
            "Ranking %d candidate(s) using profile '%s'.",
            len(raw_metrics), self._profile_name
        )

        # --- Step 1: Extract scalar value vectors for each metric -------
        raw_values = self._extract_raw_vectors(raw_metrics)

        # --- Step 2: Normalise each metric ------------------------------
        norm_vectors = self._normalise(raw_values)

        # --- Step 3: Compute composite scores ---------------------------
        candidates = self._compute_scores(raw_metrics, norm_vectors)

        # --- Step 4: Sort (ascending) with tie-breaking -----------------
        candidates = self._sort_with_tiebreak(candidates)

        # --- Step 5: Assign ranks and mark winner -----------------------
        for i, c in enumerate(candidates):
            c.rank = i + 1
        candidates[0].is_selected = True

        self._log_ranking(candidates)
        return candidates

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_raw_vectors(
        self, metrics_list: List[ModelMetrics]
    ) -> Dict[str, List[float]]:
        """
        Build per-metric lists of raw float values.

        Missing optional metrics (memory, CPU) default to ``0.0`` so
        normalisation never encounters ``None``.
        """
        return {
            "rmse":            [m.rmse for m in metrics_list],
            "mae":             [m.mae for m in metrics_list],
            "r2_penalty":      [m.r2 for m in metrics_list],   # inverted below
            "training_time":   [m.training_time for m in metrics_list],
            "prediction_time": [m.prediction_time for m in metrics_list],
            "memory_usage":    [m.memory_usage_mb or 0.0 for m in metrics_list],
            "cpu_usage":       [m.cpu_usage_pct or 0.0 for m in metrics_list],
        }

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise(
        self, raw_vectors: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Apply metric-appropriate normalisation to each vector.

        Cost metrics → min-max  (lower raw → lower normalised).
        Benefit metric (R²) → inverted min-max  (higher raw → lower penalty).
        """
        normalised: Dict[str, List[float]] = {}

        for key, values in raw_vectors.items():
            if key == "r2_penalty":
                # R² is a benefit metric; convert to penalty so that
                # *lower* is always better after normalisation.
                normalised[key] = self._invert_normalise(values)
            else:
                normalised[key] = self._minmax_normalise(values)

            self._logger.debug(
                "Normalised '%s': %s",
                key,
                [f"{v:.4f}" for v in normalised[key]],
            )

        return normalised

    @staticmethod
    def _minmax_normalise(values: List[float]) -> List[float]:
        """
        Min-max normalise a list of floats into [0, 1].
        """
        clean_values = [v for v in values if not math.isnan(v)]
        if not clean_values:
            return [1.0] * len(values)
        lo, hi = min(clean_values), max(clean_values)
        if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
            return [0.0 if not math.isnan(v) else 1.0 for v in values]
        return [((v - lo) / (hi - lo)) if not math.isnan(v) else 1.0 for v in values]

    @staticmethod
    def _invert_normalise(values: List[float]) -> List[float]:
        """
        Inverted min-max for benefit metrics (R²).
        """
        clean_values = [v for v in values if not math.isnan(v)]
        if not clean_values:
            return [1.0] * len(values)
        lo, hi = min(clean_values), max(clean_values)
        if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
            return [0.0 if not math.isnan(v) else 1.0 for v in values]
        return [((hi - v) / (hi - lo)) if not math.isnan(v) else 1.0 for v in values]

    # ------------------------------------------------------------------
    # Composite score computation
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        raw_metrics: List[ModelMetrics],
        norm_vectors: Dict[str, List[float]],
    ) -> List[CandidateResult]:
        """Attach normalised metrics and composite scores to each candidate."""
        candidates: List[CandidateResult] = []

        n = len(raw_metrics)
        for i in range(n):
            m = raw_metrics[i]

            n_rmse   = norm_vectors["rmse"][i]
            n_mae    = norm_vectors["mae"][i]
            n_r2p    = norm_vectors["r2_penalty"][i]
            n_train  = norm_vectors["training_time"][i]
            n_pred   = norm_vectors["prediction_time"][i]
            n_mem    = norm_vectors["memory_usage"][i]
            n_cpu    = norm_vectors["cpu_usage"][i]

            w = self._weights
            score = (
                w.get("rmse", 0.30)            * n_rmse
                + w.get("mae", 0.15)           * n_mae
                + w.get("r2_penalty", 0.15)    * n_r2p
                + w.get("training_time", 0.15) * n_train
                + w.get("prediction_time", 0.10) * n_pred
                + w.get("memory_usage", 0.10)  * n_mem
                + w.get("cpu_usage", 0.05)     * n_cpu
            )

            self._logger.info(
                "Composite score [%s]: %.6f "
                "(rmse=%.4f, mae=%.4f, r2p=%.4f, train=%.4f, "
                "pred=%.4f, mem=%.4f, cpu=%.4f)",
                m.model_name, score,
                n_rmse, n_mae, n_r2p, n_train, n_pred, n_mem, n_cpu,
            )

            result = CandidateResult(
                metrics=m,
                normalised_rmse=n_rmse,
                normalised_mae=n_mae,
                normalised_r2_penalty=n_r2p,
                normalised_training_time=n_train,
                normalised_prediction_time=n_pred,
                normalised_memory_mb=n_mem,
                normalised_cpu_pct=n_cpu,
                composite_score=score,
                profile_weights=dict(self._weights),
            )
            candidates.append(result)

        return candidates

    # ------------------------------------------------------------------
    # Sorting with deterministic tie-breaking
    # ------------------------------------------------------------------

    def _sort_with_tiebreak(
        self, candidates: List[CandidateResult]
    ) -> List[CandidateResult]:
        """
        Sort candidates ascending by composite score.

        Tie-breaking order (all ascending):
        1. composite_score          (primary — lower is better)
        2. raw RMSE                 (lower is better)
        3. raw MAE                  (lower is better)
        4. raw prediction_time      (lower is better)
        5. model name alphabetically (deterministic)
        """
        tol = self._tie_tolerance

        def sort_key(c: CandidateResult) -> Tuple:
            # Round composite score to tie_tolerance bucket, handle NaN
            if math.isnan(c.composite_score):
                score_bucket = float('inf')
            else:
                score_bucket = round(c.composite_score / tol) if tol > 0 else c.composite_score
                
            return (
                score_bucket,
                float('inf') if math.isnan(c.metrics.rmse) else c.metrics.rmse,
                float('inf') if math.isnan(c.metrics.mae) else c.metrics.mae,
                float('inf') if math.isnan(c.metrics.prediction_time) else c.metrics.prediction_time,
                c.metrics.model_name,
            )

        return sorted(candidates, key=sort_key)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        """
        Ensure all required weight keys are present and sum to ~1.0.

        Raises:
            SelectionStrategyError:  If weights are invalid.
        """
        missing = [k for k in self._WEIGHT_KEYS if k not in self._weights]
        if missing:
            raise SelectionStrategyError(
                f"Selection profile '{self._profile_name}' is missing "
                f"weight keys: {missing}"
            )

        total = sum(self._weights[k] for k in self._WEIGHT_KEYS)
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise SelectionStrategyError(
                f"Selection profile '{self._profile_name}' weight sum must "
                f"equal 1.0, got {total:.6f}."
            )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_ranking(self, ranked: List[CandidateResult]) -> None:
        """Log the final ranking table."""
        self._logger.info("=" * 70)
        self._logger.info(
            "Final ranking (profile: '%s'):", self._profile_name
        )
        self._logger.info(
            "  %-4s %-28s %-12s %-10s %-10s",
            "Rank", "Model", "Composite", "RMSE", "R²"
        )
        for c in ranked:
            m = c.metrics
            self._logger.info(
                "  %-4d %-28s %-12.6f %-10.6f %-10.6f %s",
                c.rank, m.model_name, c.composite_score, m.rmse, m.r2,
                " ← SELECTED" if c.is_selected else ""
            )
        self._logger.info("=" * 70)
