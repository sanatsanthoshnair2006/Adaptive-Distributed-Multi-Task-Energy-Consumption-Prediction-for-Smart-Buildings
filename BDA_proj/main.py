#!/usr/bin/env python3
"""
Adaptive Energy Prediction - Main Orchestration Entrypoint
==========================================================

Executes the complete end-to-end research pipeline, from data ingestion
to distributed model training, evaluation, and reporting.

Usage:
    python main.py [OPTIONS]
    spark-submit main.py [OPTIONS]
"""

import argparse
import os
import sys

from src.config.config_loader import ConfigLoader
from src.config.config_validator import ConfigValidator
from src.logging.logger_factory import LoggerFactory
from src.pipeline.pipeline_validator import PipelineValidator, PipelineValidationError
from src.pipeline.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Adaptive Energy Prediction pipeline."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the primary configuration file (app_config.yaml)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override the local raw dataset filename."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Provide a specific name/ID for the experiment."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing artifacts (e.g. feature store) where supported."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of all stages, overwriting existing artifacts."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Run only specific stages (e.g., '1-5' or '12'). Not yet fully supported for partial execution."
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip Explainable AI stages."
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation."
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip research report generation."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only configuration and environment validation without processing data."
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------------
    # STAGE 1: Configuration Initialization
    # ---------------------------------------------------------
    print("Initializing Configuration...")
    try:
        # ConfigLoader expects the project root, not a YAML file path.
        # It automatically loads all YAML files from <project_root>/config/.
        loader = ConfigLoader(project_root)
        config = loader.config
        
        # Apply overrides
        config["project_root"] = project_root
        if args.dataset:
            if "app" not in config: config["app"] = {}
            if "dataset" not in config["app"]: config["app"]["dataset"] = {}
            config["app"]["dataset"]["filename"] = args.dataset
            
        validator = ConfigValidator()
        validator.validate(config, project_root)
    except Exception as exc:
        print(f"CRITICAL FAILURE: Configuration initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # STAGE 2: Logging Initialization
    # ---------------------------------------------------------
    try:
        log_cfg = config.get("logging", {})
        # If absolute path not provided for logs, inject project root
        log_dir = log_cfg.get("handlers", {}).get("file", {}).get("filename", "logs/app.log")
        if not os.path.isabs(log_dir):
            abs_log_dir = os.path.join(project_root, os.path.dirname(log_dir))
            os.makedirs(abs_log_dir, exist_ok=True)
            
        LoggerFactory.configure(log_cfg, project_root)
        logger = LoggerFactory.get_logger("Main")
        logger.info("=" * 70)
        logger.info("Adaptive Energy Prediction - Pipeline Started")
        logger.info("=" * 70)
    except Exception as exc:
        print(f"CRITICAL FAILURE: Logging initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # Pipeline Validation (Dry-Run / Pre-flight)
    # ---------------------------------------------------------
    try:
        pipeline_validator = PipelineValidator(config, project_root)
        pipeline_validator.validate_all()
    except PipelineValidationError as exc:
        logger.error("Pipeline Validation Failed:\n%s", exc)
        sys.exit(1)
        
    if args.dry_run:
        logger.info("Dry-run complete. All configurations and dependencies are valid.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Pipeline Execution (Stages 3 - 20)
    # ---------------------------------------------------------
    pipeline = Pipeline(config, project_root, args)
    
    try:
        result = pipeline.run()
        if result.status == "SUCCESS":
            logger.info("Pipeline completed SUCCESSFULLY in %.2fs", result.total_duration_seconds)
            sys.exit(0)
        else:
            logger.error("Pipeline completed with status %s: %s", result.status, result.overall_error)
            sys.exit(1)
            
    except Exception as exc:
        logger.critical("Unexpected pipeline failure: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
