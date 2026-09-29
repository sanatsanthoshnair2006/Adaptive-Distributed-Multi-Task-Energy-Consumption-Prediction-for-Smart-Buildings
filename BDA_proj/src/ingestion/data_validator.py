"""
Data Validator Module
======================

Validates the integrity and schema of the loaded Spark DataFrame
before it enters the processing pipeline. Performs structural,
type, and quality checks and generates a validation report.

Design Decisions:
    - All validation runs on the Spark DataFrame (no Pandas conversion).
    - Collects all validation issues before raising, so the user
      sees every problem in one pass.
    - Produces a JSON validation report for audit and reproducibility.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.logging.logger_factory import LoggerFactory
from src.utils.file_utils import ensure_local_dir, save_json


class DataValidationError(Exception):
    """Raised when data validation fails with critical errors."""

    def __init__(self, errors: List[str], warnings: List[str]) -> None:
        self.errors = errors
        self.warnings = warnings
        message = "Data validation failed:\n"
        if errors:
            message += "ERRORS:\n" + "\n".join(f"  ✗ {e}" for e in errors)
        if warnings:
            message += "\nWARNINGS:\n" + "\n".join(
                f"  ⚠ {w}" for w in warnings
            )
        super().__init__(message)


class DataValidator:
    """
    Validates a Spark DataFrame for schema correctness, data quality,
    and structural integrity.

    Validation Checks:
        - Required columns are present
        - No unexpected duplicate columns
        - Correct delimiter was used (column count matches)
        - Data types are compatible with expected schema
        - Null percentages are within acceptable thresholds
        - No fully empty columns
        - Row count is non-zero

    Attributes:
        _expected_columns: List of required column names.
        _max_null_pct: Maximum acceptable null percentage per column.
        _report_dir: Directory to write the validation report.
    """

    # Expected data types for each column (after proper parsing)
    _EXPECTED_TYPES: Dict[str, str] = {
        "Date": "string",
        "Time": "string",
        "Global_active_power": "numeric",
        "Global_reactive_power": "numeric",
        "Voltage": "numeric",
        "Global_intensity": "numeric",
        "Sub_metering_1": "numeric",
        "Sub_metering_2": "numeric",
        "Sub_metering_3": "numeric",
    }

    # Spark types that qualify as "numeric"
    _NUMERIC_TYPES = (DoubleType, FloatType, IntegerType, LongType)

    def __init__(
        self,
        app_config: Dict[str, Any],
        project_root: str,
    ) -> None:
        """
        Initialize DataValidator.

        Args:
            app_config: The 'app' section from the merged config.
            project_root: Absolute path to the project root.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

        dataset_cfg = app_config.get("dataset", {})
        self._expected_columns: List[str] = dataset_cfg.get(
            "expected_columns", []
        )
        self._delimiter: str = dataset_cfg.get("delimiter", ";")
        self._max_null_pct: float = dataset_cfg.get(
            "max_null_percentage", 5.0
        )
        self._project_root = project_root
        self._report_dir = os.path.join(project_root, "results", "metrics")

        self._logger.info(
            "DataValidator initialized: %d expected columns, "
            "max_null_pct=%.1f%%",
            len(self._expected_columns),
            self._max_null_pct,
        )

    def validate(self, df: DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on the Spark DataFrame.

        Args:
            df: The Spark DataFrame to validate.

        Returns:
            Validation report dictionary containing all results.

        Raises:
            DataValidationError: If any critical validation checks fail.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting data validation...")
        self._logger.info("=" * 60)

        errors: List[str] = []
        warnings: List[str] = []
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "checks": {},
        }

        # ---- Check 1: Non-empty DataFrame ----
        row_count = self._check_row_count(df, errors, report)

        # ---- Check 2: Required columns present ----
        self._check_required_columns(df, errors, report)

        # ---- Check 3: Duplicate columns ----
        self._check_duplicate_columns(df, warnings, report)

        # ---- Check 4: Correct delimiter (column count) ----
        self._check_delimiter(df, errors, report)

        # ---- Check 5: Data types ----
        self._check_data_types(df, warnings, report)

        # ---- Check 6: Null percentages ----
        if row_count > 0:
            self._check_null_percentages(df, row_count, errors, warnings, report)

        # ---- Check 7: Fully empty columns ----
        if row_count > 0:
            self._check_empty_columns(df, row_count, warnings, report)

        # ---- Generate final status ----
        if errors:
            report["status"] = "FAILED"
            self._logger.error(
                "Data validation FAILED with %d error(s) and %d warning(s).",
                len(errors),
                len(warnings),
            )
        elif warnings:
            report["status"] = "PASSED_WITH_WARNINGS"
            self._logger.warning(
                "Data validation passed with %d warning(s).", len(warnings)
            )
        else:
            report["status"] = "PASSED"
            self._logger.info("Data validation PASSED — all checks OK.")

        report["errors"] = errors
        report["warnings"] = warnings
        report["error_count"] = len(errors)
        report["warning_count"] = len(warnings)

        # Persist validation report
        self._save_report(report)

        # Raise on critical errors
        if errors:
            raise DataValidationError(errors, warnings)

        return report

    # ================================================================
    # Individual Validation Checks
    # ================================================================

    def _check_row_count(
        self,
        df: DataFrame,
        errors: List[str],
        report: Dict[str, Any],
    ) -> int:
        """Check that the DataFrame contains rows."""
        row_count = df.count()
        col_count = len(df.columns)

        report["checks"]["row_count"] = {
            "rows": row_count,
            "columns": col_count,
            "status": "PASS" if row_count > 0 else "FAIL",
        }

        if row_count == 0:
            msg = "DataFrame is empty (0 rows)."
            errors.append(msg)
            self._logger.error(msg)
        else:
            self._logger.info(
                "Row count check PASSED: %d rows × %d columns",
                row_count,
                col_count,
            )

        return row_count

    def _check_required_columns(
        self,
        df: DataFrame,
        errors: List[str],
        report: Dict[str, Any],
    ) -> None:
        """Check that all expected columns are present."""
        actual_columns = set(df.columns)
        expected_set = set(self._expected_columns)

        missing = expected_set - actual_columns
        extra = actual_columns - expected_set

        report["checks"]["required_columns"] = {
            "expected": sorted(expected_set),
            "actual": sorted(actual_columns),
            "missing": sorted(missing),
            "extra": sorted(extra),
            "status": "PASS" if not missing else "FAIL",
        }

        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            errors.append(msg)
            self._logger.error(msg)
        else:
            self._logger.info(
                "Required columns check PASSED: all %d columns present.",
                len(self._expected_columns),
            )

        if extra:
            self._logger.info(
                "Extra columns detected (not an error): %s", sorted(extra)
            )

    def _check_duplicate_columns(
        self,
        df: DataFrame,
        warnings: List[str],
        report: Dict[str, Any],
    ) -> None:
        """Check for duplicate column names."""
        columns = df.columns
        seen = set()
        duplicates = set()

        for col in columns:
            if col in seen:
                duplicates.add(col)
            seen.add(col)

        report["checks"]["duplicate_columns"] = {
            "duplicates": sorted(duplicates),
            "status": "PASS" if not duplicates else "WARN",
        }

        if duplicates:
            msg = f"Duplicate column names detected: {sorted(duplicates)}"
            warnings.append(msg)
            self._logger.warning(msg)
        else:
            self._logger.info("Duplicate columns check PASSED: no duplicates.")

    def _check_delimiter(
        self,
        df: DataFrame,
        errors: List[str],
        report: Dict[str, Any],
    ) -> None:
        """
        Verify the correct delimiter was used by checking column count.

        If the wrong delimiter was used, Spark would typically produce
        a single column containing the entire row as a string.
        """
        expected_count = len(self._expected_columns)
        actual_count = len(df.columns)

        # Allow for a corrupt_record column
        is_correct = actual_count >= expected_count

        report["checks"]["delimiter"] = {
            "configured_delimiter": self._delimiter,
            "expected_column_count": expected_count,
            "actual_column_count": actual_count,
            "status": "PASS" if is_correct else "FAIL",
        }

        if not is_correct:
            msg = (
                f"Column count mismatch (possible wrong delimiter): "
                f"expected >= {expected_count}, got {actual_count}. "
                f"Configured delimiter: '{self._delimiter}'"
            )
            errors.append(msg)
            self._logger.error(msg)
        else:
            self._logger.info(
                "Delimiter check PASSED: %d columns (expected %d).",
                actual_count,
                expected_count,
            )

    def _check_data_types(
        self,
        df: DataFrame,
        warnings: List[str],
        report: Dict[str, Any],
    ) -> None:
        """
        Check that column data types match expectations.

        Numeric columns should have been inferred as numeric types
        by Spark's schema inference. String columns that should be
        numeric indicate parsing issues.
        """
        type_results: Dict[str, Dict[str, str]] = {}

        for col_name, expected_type in self._EXPECTED_TYPES.items():
            if col_name not in df.columns:
                continue

            field: StructField = df.schema[col_name]
            actual_type = field.dataType

            if expected_type == "numeric":
                is_correct = isinstance(actual_type, self._NUMERIC_TYPES)
            elif expected_type == "string":
                is_correct = isinstance(actual_type, StringType)
            else:
                is_correct = True

            type_results[col_name] = {
                "expected": expected_type,
                "actual": str(actual_type),
                "status": "PASS" if is_correct else "WARN",
            }

            if not is_correct:
                msg = (
                    f"Column '{col_name}': expected {expected_type}, "
                    f"got {actual_type}. May need type casting."
                )
                warnings.append(msg)
                self._logger.warning(msg)

        report["checks"]["data_types"] = type_results

        passed_count = sum(
            1 for r in type_results.values() if r["status"] == "PASS"
        )
        self._logger.info(
            "Data type check: %d/%d columns have expected types.",
            passed_count,
            len(type_results),
        )

    def _check_null_percentages(
        self,
        df: DataFrame,
        row_count: int,
        errors: List[str],
        warnings: List[str],
        report: Dict[str, Any],
    ) -> None:
        """
        Calculate null percentage for each column and flag those
        exceeding the configured threshold.
        """
        null_results: Dict[str, Dict[str, Any]] = {}

        # Compute null counts for all columns in a single Spark action
        null_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ]
        null_counts_row = df.select(null_exprs).collect()[0]

        for col_name in df.columns:
            null_count = null_counts_row[col_name] or 0
            null_pct = (null_count / row_count) * 100.0

            status = "PASS"
            if null_pct > self._max_null_pct:
                status = "FAIL"
                msg = (
                    f"Column '{col_name}': null percentage {null_pct:.2f}% "
                    f"exceeds threshold {self._max_null_pct:.1f}%."
                )
                errors.append(msg)
                self._logger.error(msg)
            elif null_pct > 0:
                status = "WARN"
                msg = (
                    f"Column '{col_name}': null percentage {null_pct:.2f}% "
                    f"(within threshold)."
                )
                warnings.append(msg)
                self._logger.warning(msg)

            null_results[col_name] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 4),
                "threshold": self._max_null_pct,
                "status": status,
            }

        report["checks"]["null_percentages"] = null_results

        total_nulls = sum(r["null_count"] for r in null_results.values())
        self._logger.info(
            "Null check complete: %d total nulls across %d columns.",
            total_nulls,
            len(null_results),
        )

    def _check_empty_columns(
        self,
        df: DataFrame,
        row_count: int,
        warnings: List[str],
        report: Dict[str, Any],
    ) -> None:
        """Check for columns that are entirely null."""
        empty_columns: List[str] = []

        # Reuse null counts from previous check if available
        null_check = report.get("checks", {}).get("null_percentages", {})

        for col_name in df.columns:
            col_data = null_check.get(col_name, {})
            null_count = col_data.get("null_count", None)

            if null_count is None:
                # Fallback: compute directly
                null_count = df.filter(F.col(col_name).isNull()).count()

            if null_count == row_count:
                empty_columns.append(col_name)

        report["checks"]["empty_columns"] = {
            "empty_columns": empty_columns,
            "status": "PASS" if not empty_columns else "WARN",
        }

        if empty_columns:
            msg = f"Fully empty columns detected: {empty_columns}"
            warnings.append(msg)
            self._logger.warning(msg)
        else:
            self._logger.info("Empty columns check PASSED: no fully empty columns.")

    # ================================================================
    # Report Persistence
    # ================================================================

    def _save_report(self, report: Dict[str, Any]) -> None:
        """
        Save the validation report as a JSON file.

        Args:
            report: Validation report dictionary.
        """
        try:
            report_path = os.path.join(
                self._report_dir, "validation_report.json"
            )
            save_json(report, report_path)
            self._logger.info("Validation report saved: %s", report_path)
        except Exception as e:
            self._logger.warning(
                "Failed to save validation report: %s", str(e)
            )

    # ================================================================
    # Public Helpers
    # ================================================================

    def print_summary(self, df: DataFrame) -> None:
        """
        Print a human-readable summary of the DataFrame to the log.

        Args:
            df: Spark DataFrame to summarize.
        """
        self._logger.info("=" * 60)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 60)
        self._logger.info("Columns: %s", df.columns)
        self._logger.info("Schema:")
        for field in df.schema.fields:
            self._logger.info(
                "  %-30s  %s  (nullable=%s)",
                field.name,
                field.dataType,
                field.nullable,
            )
        self._logger.info("Row count: %d", df.count())
        self._logger.info("=" * 60)
