"""
Models Package
==============

Composite Adaptive Model Selection framework.

Public API
----------
- ``ModelMetrics``             — raw evaluation metrics dataclass
- ``CandidateResult``          — normalised + ranked candidate dataclass
- ``FailedCandidateResult``    — sentinel for failed model training
- ``ModelFactory``             — creates configured Spark MLlib estimators
- ``ModelFactoryError``        — raised by ModelFactory
- ``ModelManager``             — saves / loads Spark MLlib models in HDFS
- ``ModelManagerError``        — raised by ModelManager
- ``SelectionStrategy``        — composite score normalisation and ranking
- ``SelectionStrategyError``   — raised by SelectionStrategy
- ``CompositeAdaptiveSelector``— full selection orchestrator
- ``AllModelsFailedError``     — raised if all candidates fail for a task
- ``ExperimentManager``        — persists experiment records
"""

from src.models.model_metrics import (
    ModelMetrics,
    CandidateResult,
    FailedCandidateResult,
)
from src.models.model_factory import ModelFactory, ModelFactoryError
from src.models.model_manager import ModelManager, ModelManagerError
from src.models.selection_strategy import SelectionStrategy, SelectionStrategyError
from src.models.composite_adaptive_selector import (
    CompositeAdaptiveSelector,
    AllModelsFailedError,
    AdaptiveSelectorError,
    ExperimentManager,
)
from src.models.cross_validator import TimeSeriesCrossValidator, CrossValidatorError
from src.models.hyperparameter_tuner import HyperparameterTuner, HyperparameterTunerError
from src.models.model_trainer import ModelTrainer, ModelTrainerError
from src.models.training_pipeline import TrainingPipeline, TrainingPipelineError

__all__ = [
    # Metrics
    "ModelMetrics",
    "CandidateResult",
    "FailedCandidateResult",
    # Factory
    "ModelFactory",
    "ModelFactoryError",
    # Manager
    "ModelManager",
    "ModelManagerError",
    # Strategy
    "SelectionStrategy",
    "SelectionStrategyError",
    # Orchestrator
    "CompositeAdaptiveSelector",
    "AllModelsFailedError",
    "AdaptiveSelectorError",
    "ExperimentManager",
    # Validation & Tuning
    "TimeSeriesCrossValidator",
    "CrossValidatorError",
    "HyperparameterTuner",
    "HyperparameterTunerError",
    # Training Pipeline
    "ModelTrainer",
    "ModelTrainerError",
    "TrainingPipeline",
    "TrainingPipelineError",
]
