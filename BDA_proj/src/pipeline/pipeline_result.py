"""
Pipeline Result Module
======================

Data structures for tracking the results of individual stages
and the overall pipeline execution.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Represents the execution outcome of a single pipeline stage."""
    stage_id: int
    stage_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    status: str = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED, SKIPPED
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Represents the complete execution outcome of the pipeline."""
    execution_id: str
    experiment_id: Optional[str] = None
    start_time: str = ""
    end_time: Optional[str] = None
    total_duration_seconds: float = 0.0
    status: str = "INITIALIZED"  # INITIALIZED, RUNNING, SUCCESS, PARTIAL_SUCCESS, FAILED
    stages: List[StageResult] = field(default_factory=list)
    configuration_profile: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    overall_error: Optional[str] = None

    def get_stage(self, stage_name: str) -> Optional[StageResult]:
        for s in self.stages:
            if s.stage_name == stage_name:
                return s
        return None

    def add_stage(self, stage: StageResult) -> None:
        self.stages.append(stage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        """Save the pipeline result to a local JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
