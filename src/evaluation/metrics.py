"""Evaluation Metrics Module.

Calculates Precision, Recall, and F1-Score for fraud detection
by comparing detected clusters against ground truth labels.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""
    
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def __str__(self) -> str:
        """Format results as a readable string."""
        return (
            f"Evaluation Results:\n"
            f"  True Positives:  {self.true_positives}\n"
            f"  False Positives: {self.false_positives}\n"
            f"  True Negatives:  {self.true_negatives}\n"
            f"  False Negatives: {self.false_negatives}\n"
            f"  --------------------------------\n"
            f"  Precision:       {self.precision:.4f} ({self.precision:.1%})\n"
            f"  Recall:          {self.recall:.4f} ({self.recall:.1%})\n"
            f"  F1-Score:        {self.f1_score:.4f} ({self.f1_score:.1%})\n"
            f"  Accuracy:        {self.accuracy:.4f} ({self.accuracy:.1%})"
        )


class FraudMetrics:
    """
    Evaluates fraud detection performance against ground truth.
    
    Compares the detected_fraud column (from clustering) against
    the is_fraud column (ground truth from data generation).
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self._results: Optional[EvaluationResults] = None
    
    @staticmethod
    def calculate_precision(tp: int, fp: int) -> float:
        """
        Calculate precision: TP / (TP + FP)
        
        Precision answers: Of all records flagged as fraud,
        what fraction were actually fraud?
        """
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def calculate_recall(tp: int, fn: int) -> float:
        """
        Calculate recall: TP / (TP + FN)
        
        Recall answers: Of all actual fraud cases,
        what fraction did we detect?
        """
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """
        Calculate F1-Score: 2 * (precision * recall) / (precision + recall)
        
        F1 is the harmonic mean of precision and recall.
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calculate accuracy: (TP + TN) / Total
        
        Overall correctness of predictions.
        """
        total = tp + tn + fp + fn
        if total == 0:
            return 0.0
        return (tp + tn) / total
    
    def evaluate(
        self, 
        df: pd.DataFrame,
        ground_truth_col: str = "is_fraud",
        prediction_col: str = "detected_fraud"
    ) -> EvaluationResults:
        """
        Evaluate fraud detection against ground truth.
        
        Args:
            df: DataFrame with both ground truth and predictions.
            ground_truth_col: Column name for actual fraud labels.
            prediction_col: Column name for predicted fraud labels.
            
        Returns:
            EvaluationResults with all metrics.
        """
        if ground_truth_col not in df.columns:
            raise ValueError(f"Ground truth column '{ground_truth_col}' not found")
        if prediction_col not in df.columns:
            raise ValueError(f"Prediction column '{prediction_col}' not found")
        
        # Convert to boolean
        y_true = df[ground_truth_col].astype(bool)
        y_pred = df[prediction_col].astype(bool)
        
        # Calculate confusion matrix elements
        tp = int(((y_true == True) & (y_pred == True)).sum())
        fp = int(((y_true == False) & (y_pred == True)).sum())
        tn = int(((y_true == False) & (y_pred == False)).sum())
        fn = int(((y_true == True) & (y_pred == False)).sum())
        
        # Calculate metrics
        precision = self.calculate_precision(tp, fp)
        recall = self.calculate_recall(tp, fn)
        f1 = self.calculate_f1(precision, recall)
        accuracy = self.calculate_accuracy(tp, tn, fp, fn)
        
        self._results = EvaluationResults(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy
        )
        
        return self._results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get the confusion matrix from the last evaluation.
        
        Returns:
            2x2 numpy array [[TN, FP], [FN, TP]]
        """
        if self._results is None:
            raise ValueError("No evaluation has been performed yet")
        
        return np.array([
            [self._results.true_negatives, self._results.false_positives],
            [self._results.false_negatives, self._results.true_positives]
        ])
    
    def get_detailed_report(
        self, 
        df: pd.DataFrame,
        ground_truth_col: str = "is_fraud",
        prediction_col: str = "detected_fraud"
    ) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            df: DataFrame with results.
            ground_truth_col: Ground truth column name.
            prediction_col: Prediction column name.
            
        Returns:
            Formatted report string.
        """
        results = self.evaluate(df, ground_truth_col, prediction_col)
        
        # Calculate by fraud type if available
        fraud_type_analysis = ""
        if "fraud_type" in df.columns:
            fraud_types = df[df[ground_truth_col] == True]["fraud_type"].value_counts()
            detected_by_type = df[
                (df[ground_truth_col] == True) & (df[prediction_col] == True)
            ]["fraud_type"].value_counts()
            
            fraud_type_analysis = "\nDetection Rate by Fraud Type:\n"
            fraud_type_analysis += "-" * 40 + "\n"
            for fraud_type in fraud_types.index:
                if pd.notna(fraud_type):
                    total = fraud_types.get(fraud_type, 0)
                    detected = detected_by_type.get(fraud_type, 0)
                    rate = detected / total if total > 0 else 0
                    fraud_type_analysis += f"  {fraud_type}: {detected}/{total} ({rate:.1%})\n"
        
        report = f"""
{'=' * 60}
FRAUD DETECTION EVALUATION REPORT
{'=' * 60}

Dataset Statistics:
  Total Records:        {len(df)}
  Actual Frauds:        {df[ground_truth_col].sum()}
  Detected as Fraud:    {df[prediction_col].sum()}

Confusion Matrix:
                    Predicted
                    Neg     Pos
  Actual  Neg       {results.true_negatives:<7} {results.false_positives}
          Pos       {results.false_negatives:<7} {results.true_positives}

Performance Metrics:
  Precision:  {results.precision:.4f} ({results.precision:.1%})
              -> Of flagged records, {results.precision:.1%} were actual fraud
  
  Recall:     {results.recall:.4f} ({results.recall:.1%})
              -> Of actual frauds, {results.recall:.1%} were detected
  
  F1-Score:   {results.f1_score:.4f} ({results.f1_score:.1%})
              -> Harmonic mean of precision and recall
  
  Accuracy:   {results.accuracy:.4f} ({results.accuracy:.1%})
              -> Overall classification correctness
{fraud_type_analysis}
{'=' * 60}
"""
        return report


if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(10)],
        "is_fraud": [False, True, True, False, True, False, False, True, False, True],
        "detected_fraud": [False, True, False, False, True, True, False, True, False, True],
        "fraud_type": [None, "typo", "shared_iban", None, "near_dup", None, None, "typo", None, "typo"],
    })
    
    metrics = FraudMetrics()
    report = metrics.get_detailed_report(test_df)
    print(report)
