"""
Model Factory Module
=====================

Centralised factory for creating configured Spark MLlib estimators.

Design Decisions:
    - Open/Closed Principle:  New models can be added by registering a new
      builder method decorated with ``@_register``.  No existing code changes
      required.
    - Configuration-driven:  Every hyperparameter is read from
      ``model_config.yaml``.  Nothing is hardcoded inside Python.
    - Spark MLlib only:  No scikit-learn, no external frameworks.
    - Returns estimators, not fitted models:  The factory does not train;
      it only constructs configured Spark ML Estimator objects ready for
      ``fit()``.
    - Disabled models:  If ``enabled: false`` in config, the factory skips
      that model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from pyspark.ml import Estimator
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)

from src.logging.logger_factory import LoggerFactory


class ModelFactoryError(Exception):
    """Raised when model construction fails."""


class ModelFactory:
    """
    Factory for creating configured Spark MLlib regression estimators.

    Each candidate model is built from the ``model_config.yaml`` section
    (``config["model"]["models"]``).

    Usage::

        factory = ModelFactory(config)
        candidates = factory.get_enabled_candidates()
        # candidates → [("linear_regression", <LinearRegression>), ...]

    Args:
        config (Dict[str, Any]):  Full merged configuration dictionary.
        features_col (str):  Name of the assembled features vector column.
                             Defaults to ``"features"``.
        label_col (str):     Name of the label (target) column.
                             Defaults to ``"label"``.
    """

    # Internal registry: model_key → builder callable
    _BUILDERS: Dict[str, Callable[..., Estimator]] = {}

    def __init__(
        self,
        config: Dict[str, Any],
        features_col: str = "features",
        label_col: str = "label",
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._model_cfg: Dict[str, Any] = (
            config.get("model", {}).get("models", {})
        )
        self._features_col = features_col
        self._label_col = label_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_enabled_candidates(self) -> List[Tuple[str, Estimator]]:
        """
        Return (model_name, estimator) tuples for all enabled models.

        Returns:
            List of ``(model_name, Spark Estimator)`` pairs, ordered as
            they appear in ``model_config.yaml``.

        Raises:
            ModelFactoryError:  If a model key has no registered builder.
        """
        candidates: List[Tuple[str, Estimator]] = []

        for key in self._BUILDERS:
            model_params = self._model_cfg.get(key, {})
            if not model_params.get("enabled", True):
                self._logger.info(
                    "Model '%s' is disabled in config — skipping.", key
                )
                continue
            try:
                estimator = self._build(key)
                candidates.append((key, estimator))
                self._logger.info("Built candidate model: '%s'", key)
            except Exception as exc:
                raise ModelFactoryError(
                    f"Failed to build model '{key}': {exc}"
                ) from exc

        if not candidates:
            raise ModelFactoryError(
                "No candidate models are enabled in model_config.yaml."
            )

        return candidates

    def build_single(self, model_name: str) -> Estimator:
        """
        Build and return one specific model by name.

        Args:
            model_name (str):  Key as it appears in ``model_config.yaml``.

        Returns:
            Configured Spark ML Estimator.

        Raises:
            ModelFactoryError:  If the name is unknown or build fails.
        """
        if model_name not in self._BUILDERS:
            raise ModelFactoryError(
                f"Unknown model name: '{model_name}'. "
                f"Available: {list(self._BUILDERS.keys())}"
            )
        return self._build(model_name)

    def list_available(self) -> List[str]:
        """Return names of all registered (not necessarily enabled) models."""
        return list(self._BUILDERS.keys())

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _build(self, key: str) -> Estimator:
        """Dispatch to the appropriate builder."""
        return self._BUILDERS[key](self, key)

    # ------------------------------------------------------------------
    # Builders — one per model family
    # ------------------------------------------------------------------

    def _build_linear_regression(self, key: str) -> LinearRegression:
        """Build a configured LinearRegression estimator."""
        p = self._model_cfg.get(key, {})
        return LinearRegression(
            featuresCol=self._features_col,
            labelCol=self._label_col,
            predictionCol="prediction",
            maxIter=int(p.get("max_iter", 100)),
            regParam=float(p.get("reg_param", 0.0)),
            elasticNetParam=float(p.get("elastic_net_param", 0.0)),
            standardization=bool(p.get("standardization", True)),
        )

    def _build_decision_tree(self, key: str) -> DecisionTreeRegressor:
        """Build a configured DecisionTreeRegressor estimator."""
        p = self._model_cfg.get(key, {})
        return DecisionTreeRegressor(
            featuresCol=self._features_col,
            labelCol=self._label_col,
            predictionCol="prediction",
            maxDepth=int(p.get("max_depth", 5)),
            minInstancesPerNode=int(p.get("min_instances_per_node", 1)),
            seed=int(p.get("seed", 42)),
        )

    def _build_random_forest(self, key: str) -> RandomForestRegressor:
        """Build a configured RandomForestRegressor estimator."""
        p = self._model_cfg.get(key, {})
        return RandomForestRegressor(
            featuresCol=self._features_col,
            labelCol=self._label_col,
            predictionCol="prediction",
            numTrees=int(p.get("num_trees", 100)),
            maxDepth=int(p.get("max_depth", 5)),
            minInstancesPerNode=int(p.get("min_instances_per_node", 1)),
            featureSubsetStrategy=str(p.get("feature_subset_strategy", "auto")),
            seed=int(p.get("seed", 42)),
        )

    def _build_gradient_boosted_tree(self, key: str) -> GBTRegressor:
        """Build a configured GBTRegressor estimator."""
        p = self._model_cfg.get(key, {})
        return GBTRegressor(
            featuresCol=self._features_col,
            labelCol=self._label_col,
            predictionCol="prediction",
            maxIter=int(p.get("max_iter", 50)),
            maxDepth=int(p.get("max_depth", 5)),
            stepSize=float(p.get("step_size", 0.1)),
            subsamplingRate=float(p.get("subsample_rate", 1.0)),
            seed=int(p.get("seed", 42)),
        )


# ---------------------------------------------------------------------------
# Register all built-in builders after class definition
# ---------------------------------------------------------------------------

ModelFactory._BUILDERS = {
    "linear_regression":   ModelFactory._build_linear_regression,
    "decision_tree":       ModelFactory._build_decision_tree,
    "random_forest":       ModelFactory._build_random_forest,
    "gradient_boosted_tree": ModelFactory._build_gradient_boosted_tree,
}
