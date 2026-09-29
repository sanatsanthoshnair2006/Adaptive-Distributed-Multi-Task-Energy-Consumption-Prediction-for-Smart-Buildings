"""
Statistical Tests Module
=========================

Implements statistical validation of model performance differences.

Design Decisions:
    - Operates on paired errors (Model A vs Model B) aligned by Timestamp.
    - Implements Paired t-test and Wilcoxon Signed-Rank Test.
    - Supports multiple comparison correction (Bonferroni).
    - Extracts data to Pandas/SciPy for actual statistical computation, as Spark MLlib
      does not have native paired non-parametric tests.
    - Implements sampling if the dataset is too large to fit in driver memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy import stats
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.logging.logger_factory import LoggerFactory


@dataclass
class StatisticalTestResult:
    """Stores the result of a statistical comparison between two models."""
    baseline_model: str
    candidate_model: str
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    is_significant: bool
    correction_method: str
    sample_size: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    notes: str = ""


class StatisticalTests:
    """
    Executes statistical tests to compare model prediction errors.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        timestamp_col (str): Column used to align predictions. Defaults to "Timestamp".
        label_col (str): Ground truth column name. Defaults to "label".
        prediction_col (str): Prediction column name. Defaults to "prediction".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        timestamp_col: str = "Timestamp",
        label_col: str = "label",
        prediction_col: str = "prediction"
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._timestamp_col = timestamp_col
        self._label_col = label_col
        self._prediction_col = prediction_col

        eval_cfg = config.get("evaluation", {})
        stat_cfg = eval_cfg.get("statistics", {})
        
        self._alpha = float(stat_cfg.get("alpha", 0.05))
        self._test_type = stat_cfg.get("test_type", "wilcoxon").lower()
        self._correction = stat_cfg.get("multiple_comparison_correction", "none").lower()
        
        ci_cfg = stat_cfg.get("confidence_intervals", {})
        self._compute_ci = str(ci_cfg.get("enabled", "true")).lower() == "true"
        self._max_sample_size = int(ci_cfg.get("max_sample_size", 100000))

    def compare_models(
        self,
        baseline_name: str,
        baseline_df: DataFrame,
        candidate_name: str,
        candidate_df: DataFrame,
        num_comparisons: int = 1
    ) -> StatisticalTestResult:
        """
        Compare two models by aligning their predictions and testing their paired errors.

        Args:
            baseline_name: Identifier for the baseline model.
            baseline_df: Predictions DataFrame for the baseline model.
            candidate_name: Identifier for the candidate model.
            candidate_df: Predictions DataFrame for the candidate model.
            num_comparisons: Total number of comparisons being made (for Bonferroni).

        Returns:
            StatisticalTestResult: The results of the statistical test.
        """
        self._logger.info("Comparing %s (baseline) vs %s (candidate)...", baseline_name, candidate_name)

        # 1. Align the datasets on Timestamp to ensure strict pairing
        # Rename columns to avoid collisions
        base_renamed = baseline_df.select(
            F.col(self._timestamp_col),
            F.col(self._label_col),
            F.col(self._prediction_col).alias("base_pred")
        )
        cand_renamed = candidate_df.select(
            F.col(self._timestamp_col),
            F.col(self._prediction_col).alias("cand_pred")
        )

        joined_df = base_renamed.join(cand_renamed, on=self._timestamp_col, how="inner")
        
        # 2. Compute absolute errors
        # Error = |y - y_pred|
        errors_df = joined_df.select(
            F.abs(F.col(self._label_col) - F.col("base_pred")).alias("err_base"),
            F.abs(F.col(self._label_col) - F.col("cand_pred")).alias("err_cand")
        )

        total_pairs = errors_df.count()
        if total_pairs < 30:
            msg = f"Insufficient paired observations ({total_pairs} < 30) for reliable statistical testing."
            self._logger.warning(msg)
            return StatisticalTestResult(
                baseline_model=baseline_name,
                candidate_model=candidate_name,
                test_name=self._test_type,
                statistic=0.0,
                p_value=1.0,
                alpha=self._alpha,
                is_significant=False,
                correction_method="none",
                sample_size=total_pairs,
                notes=msg
            )

        # 3. Sample if necessary to fit in driver memory
        if self._max_sample_size > 0 and total_pairs > self._max_sample_size:
            fraction = self._max_sample_size / total_pairs
            self._logger.info("Sampling %.2f%% of paired errors for stats testing.", fraction * 100)
            errors_df = errors_df.sample(withReplacement=False, fraction=fraction, seed=42)

        # 4. Collect to driver (Pandas/NumPy)
        pdf = errors_df.toPandas()
        err_base = pdf["err_base"].to_numpy()
        err_cand = pdf["err_cand"].to_numpy()
        
        # 5. Apply multiple comparison correction
        adjusted_alpha = self._alpha
        if self._correction == "bonferroni" and num_comparisons > 1:
            adjusted_alpha = self._alpha / num_comparisons
            self._logger.debug(
                "Applied Bonferroni correction: alpha %.4f -> %.4f (n=%d)", 
                self._alpha, adjusted_alpha, num_comparisons
            )

        # 6. Execute chosen statistical test
        stat_val = 0.0
        p_val = 1.0
        ci_lower = None
        ci_upper = None
        notes = ""

        try:
            if self._test_type == "wilcoxon":
                # Wilcoxon signed-rank test for paired non-parametric data
                res = stats.wilcoxon(err_base, err_cand, zero_method='zsplit')
                stat_val = float(res.statistic)
                p_val = float(res.pvalue)
            
            elif self._test_type == "paired_t":
                # Paired t-test
                res = stats.ttest_rel(err_base, err_cand)
                stat_val = float(res.statistic)
                p_val = float(res.pvalue)
            
            else:
                notes = f"Unknown test_type '{self._test_type}'. Falling back to defaults."
                self._logger.warning(notes)
                
            # Compute basic confidence interval for the mean paired difference
            if self._compute_ci and len(err_base) > 1:
                diffs = err_cand - err_base
                mean_diff = np.mean(diffs)
                sem_diff = stats.sem(diffs)
                
                if sem_diff > 0:
                    # using t-distribution for CI
                    dof = len(diffs) - 1
                    ci = stats.t.interval(1 - adjusted_alpha, dof, loc=mean_diff, scale=sem_diff)
                    ci_lower, ci_upper = float(ci[0]), float(ci[1])
                else:
                    ci_lower, ci_upper = mean_diff, mean_diff
                
        except Exception as exc:
            notes = f"Statistical test failed: {exc}"
            self._logger.error("[%s vs %s] %s", baseline_name, candidate_name, notes)

        is_significant = bool(p_val < adjusted_alpha)
        
        # Log interpretation
        if is_significant:
            self._logger.info("Statistically significant difference detected (p=%.4e < %.4f).", p_val, adjusted_alpha)
        else:
            self._logger.info("No statistically significant difference (p=%.4e >= %.4f).", p_val, adjusted_alpha)

        return StatisticalTestResult(
            baseline_model=baseline_name,
            candidate_model=candidate_name,
            test_name=self._test_type,
            statistic=stat_val,
            p_value=p_val,
            alpha=adjusted_alpha,
            is_significant=is_significant,
            correction_method=self._correction,
            sample_size=len(pdf),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            notes=notes
        )
