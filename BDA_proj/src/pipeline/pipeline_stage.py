"""
Pipeline Stage Module
======================

Provides the base tracking logic and wrapper for executing a pipeline stage.
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.logging.logger_factory import LoggerFactory
from src.pipeline.pipeline_result import StageResult


class PipelineStageError(Exception):
    """Raised when a critical stage fails."""
    pass


class PipelineStage:
    """
    Executes and tracks a single logical stage in the pipeline.
    
    Handles timing, status updates, and logging consistently.
    """

    def __init__(
        self,
        stage_id: int,
        stage_name: str,
        is_critical: bool = True,
        skip_if: bool = False
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.is_critical = is_critical
        self.skip_if = skip_if

    def execute(
        self,
        action: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> StageResult:
        """
        Execute the provided callable within the stage tracking context.
        """
        result = StageResult(
            stage_id=self.stage_id,
            stage_name=self.stage_name,
            start_time=datetime.now().isoformat()
        )

        if self.skip_if:
            self._logger.info(
                "[%02d] Stage '%s' is SKIPPED.", self.stage_id, self.stage_name
            )
            result.status = "SKIPPED"
            result.end_time = datetime.now().isoformat()
            return result

        self._logger.info("=" * 60)
        self._logger.info("[%02d] STARTING STAGE: %s", self.stage_id, self.stage_name)
        self._logger.info("=" * 60)

        t0 = time.perf_counter()
        result.status = "RUNNING"

        try:
            # Execute the actual logic
            output = action(*args, **kwargs)
            
            # Store the output if it's a dict (useful for artifacts mapping)
            if isinstance(output, dict):
                result.artifacts = output
                
            result.status = "SUCCESS"
            
        except Exception as exc:
            result.status = "FAILED"
            result.error_message = str(exc)
            
            if self.is_critical:
                self._logger.error(
                    "[%02d] CRITICAL FAILURE in '%s': %s",
                    self.stage_id, self.stage_name, exc, exc_info=True
                )
                result.end_time = datetime.now().isoformat()
                result.duration_seconds = time.perf_counter() - t0
                raise PipelineStageError(f"Stage {self.stage_name} failed: {exc}") from exc
            else:
                self._logger.warning(
                    "[%02d] NON-CRITICAL FAILURE in '%s': %s",
                    self.stage_id, self.stage_name, exc
                )
                
        finally:
            t1 = time.perf_counter()
            result.duration_seconds = t1 - t0
            result.end_time = datetime.now().isoformat()
            
            self._logger.info(
                "[%02d] COMPLETED STAGE: %s (Status: %s, Time: %.2fs)",
                self.stage_id, self.stage_name, result.status, result.duration_seconds
            )

        return result
