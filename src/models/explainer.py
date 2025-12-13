"""
Fraud Explainability Module.

Provides explanations for fraud detection decisions using SHAP
and custom interpretability methods.
"""

from typing import Optional, Any

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .base import BaseDetector
from .features import FeatureExtractor


class FraudExplainer:
    """
    Explains fraud detection predictions.
    
    Provides various explanation methods:
    - SHAP values for tree-based models
    - Feature contribution analysis
    - Cluster-based explanations for DBSCAN
    - Similarity breakdowns
    """
    
    def __init__(self, detector: BaseDetector):
        """
        Initialize the explainer.
        
        Args:
            detector: The fraud detector to explain.
        """
        self.detector = detector
        self._shap_explainer: Optional[Any] = None
        self._feature_extractor = FeatureExtractor()
    
    def explain_isolation_forest(
        self, 
        df: pd.DataFrame, 
        idx: int,
        num_features: int = 10
    ) -> dict:
        """
        Explain Isolation Forest prediction using SHAP.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of record to explain.
            num_features: Number of top features to show.
            
        Returns:
            Dictionary with SHAP-based explanation.
        """
        if not HAS_SHAP:
            return self._explain_feature_deviation(df, idx, num_features)
        
        # Get the model from detector
        model = getattr(self.detector, '_model', None)
        if model is None:
            return self._explain_feature_deviation(df, idx, num_features)
        
        # Extract features
        feature_df = self._feature_extractor.extract_features(df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        X = numeric_features.fillna(0).values
        
        # Scale if scaler exists
        scaler = getattr(self.detector, '_scaler', None)
        if scaler is not None:
            X = scaler.transform(X)
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[idx:idx+1])
            
            # Get feature contributions
            feature_names = list(numeric_features.columns)
            contributions = dict(zip(feature_names, shap_values[0]))
            
            # Sort by absolute contribution
            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:num_features]
            )
            
            return {
                "method": "shap",
                "base_value": float(explainer.expected_value),
                "feature_contributions": sorted_contributions,
                "prediction_score": float(self.detector.predict(df.iloc[[idx]])["score"].iloc[0])
            }
        except Exception as e:
            # Fall back to feature deviation
            return self._explain_feature_deviation(df, idx, num_features)
    
    def _explain_feature_deviation(
        self, 
        df: pd.DataFrame, 
        idx: int,
        num_features: int = 10
    ) -> dict:
        """
        Explain using feature deviation from mean.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of record to explain.
            num_features: Number of top features to show.
            
        Returns:
            Dictionary with deviation-based explanation.
        """
        feature_df = self._feature_extractor.extract_features(df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        
        # Calculate z-scores
        X = numeric_features.fillna(0)
        means = X.mean()
        stds = X.std().replace(0, 1)
        
        z_scores = (X.iloc[idx] - means) / stds
        
        # Sort by absolute z-score
        sorted_features = z_scores.abs().sort_values(ascending=False)[:num_features]
        
        contributions = {
            feature: {
                "z_score": float(z_scores[feature]),
                "value": float(X.iloc[idx][feature]),
                "mean": float(means[feature]),
                "std": float(stds[feature])
            }
            for feature in sorted_features.index
        }
        
        return {
            "method": "feature_deviation",
            "feature_contributions": contributions,
            "prediction_score": float(self.detector.predict(df.iloc[[idx]])["score"].iloc[0])
        }
    
    def explain_dbscan(
        self, 
        df: pd.DataFrame, 
        idx: int
    ) -> dict:
        """
        Explain DBSCAN clustering decision.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of record to explain.
            
        Returns:
            Dictionary with cluster-based explanation.
        """
        # Get cluster information
        labels = getattr(self.detector, '_labels', None)
        clusters = getattr(self.detector, '_clusters', [])
        
        if labels is None:
            return {
                "error": "DBSCAN not fitted",
                "index": idx
            }
        
        cluster_id = labels[idx] if idx < len(labels) else -1
        
        if cluster_id == -1:
            return {
                "index": idx,
                "cluster_id": -1,
                "is_noise": True,
                "explanation": "Record does not belong to any cluster (noise point)"
            }
        
        # Find the cluster
        cluster_info = None
        for cluster in clusters:
            if cluster.cluster_id == cluster_id:
                cluster_info = cluster
                break
        
        if cluster_info is None:
            return {
                "index": idx,
                "cluster_id": cluster_id,
                "explanation": "Cluster information not available"
            }
        
        # Get field-level similarity breakdown
        field_contributions = {}
        if hasattr(self.detector, 'get_field_contributions'):
            field_contributions = self.detector.get_field_contributions(df, idx)
        
        # Get other cluster members
        other_members = [i for i in cluster_info.record_indices if i != idx][:5]
        member_details = []
        for member_idx in other_members:
            if member_idx < len(df):
                member_details.append({
                    "index": member_idx,
                    "customer_id": df.iloc[member_idx].get("customer_id", "N/A")
                })
        
        return {
            "index": idx,
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_info.record_indices),
            "similarity_score": cluster_info.similarity_score,
            "detection_reason": cluster_info.detection_reason,
            "field_contributions": field_contributions,
            "similar_records": member_details,
            "explanation": f"Record belongs to cluster of {len(cluster_info.record_indices)} similar records"
        }
    
    def explain_ensemble(
        self, 
        df: pd.DataFrame, 
        idx: int
    ) -> dict:
        """
        Explain ensemble prediction by breaking down individual detector contributions.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of record to explain.
            
        Returns:
            Dictionary with ensemble breakdown.
        """
        if not hasattr(self.detector, 'detectors'):
            return self.detector.explain(df, idx)
        
        single_df = df.iloc[[idx]] if isinstance(idx, int) else df.loc[[idx]]
        ensemble_pred = self.detector.predict(single_df)
        
        detector_explanations = {}
        for detector, weight in self.detector.detectors:
            preds = detector.predict(single_df)
            
            # Get detailed explanation from individual detector
            try:
                detailed = detector.explain(df, idx)
            except Exception:
                detailed = {}
            
            detector_explanations[detector.name] = {
                "score": float(preds["score"].iloc[0]),
                "is_fraud": bool(preds["is_fraud"].iloc[0]),
                "weight": weight,
                "contribution": float(preds["score"].iloc[0]) * weight,
                "reason": preds["reason"].iloc[0],
                "details": detailed.get("feature_contributions", {})
            }
        
        # Calculate weighted contribution
        total_weight = sum(w for _, w in self.detector.detectors)
        
        return {
            "index": idx,
            "ensemble_score": float(ensemble_pred["score"].iloc[0]),
            "ensemble_fraud": bool(ensemble_pred["is_fraud"].iloc[0]),
            "strategy": getattr(self.detector, 'strategy', 'unknown'),
            "threshold": getattr(self.detector, '_threshold', 0.5),
            "detector_breakdown": detector_explanations,
            "total_weight": total_weight
        }
    
    def get_global_feature_importance(
        self, 
        df: pd.DataFrame,
        num_features: int = 20
    ) -> dict:
        """
        Get global feature importance across the dataset.
        
        Args:
            df: DataFrame with customer records.
            num_features: Number of top features to return.
            
        Returns:
            Dictionary with feature importance scores.
        """
        model = getattr(self.detector, '_model', None)
        
        if model is None:
            return {"error": "Model not available"}
        
        # Extract features
        feature_df = self._feature_extractor.extract_features(df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        X = numeric_features.fillna(0).values
        
        # Scale if needed
        scaler = getattr(self.detector, '_scaler', None)
        if scaler is not None:
            X = scaler.transform(X)
        
        feature_names = list(numeric_features.columns)
        
        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Calculate mean absolute SHAP value per feature
                importance = np.abs(shap_values).mean(axis=0)
                importance_dict = dict(zip(feature_names, importance))
                
                sorted_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:num_features]
                )
                
                return {
                    "method": "shap",
                    "feature_importance": sorted_importance
                }
            except Exception:
                pass
        
        # Fallback: variance-based importance
        variances = numeric_features.var()
        sorted_variance = variances.sort_values(ascending=False)[:num_features]
        
        return {
            "method": "variance",
            "feature_importance": sorted_variance.to_dict()
        }
    
    def generate_report(
        self, 
        df: pd.DataFrame, 
        flagged_indices: list[int]
    ) -> str:
        """
        Generate a human-readable explanation report for flagged records.
        
        Args:
            df: DataFrame with customer records.
            flagged_indices: List of flagged record indices.
            
        Returns:
            Formatted report string.
        """
        report_lines = [
            "=" * 60,
            "FRAUD DETECTION EXPLANATION REPORT",
            "=" * 60,
            f"\nTotal flagged records: {len(flagged_indices)}\n"
        ]
        
        for idx in flagged_indices[:20]:  # Limit to 20 records
            report_lines.append("-" * 40)
            report_lines.append(f"Record Index: {idx}")
            
            if "customer_id" in df.columns:
                report_lines.append(f"Customer ID: {df.iloc[idx]['customer_id']}")
            
            # Get explanation
            explanation = self.detector.explain(df, idx)
            
            report_lines.append(f"Fraud Score: {explanation.get('score', 'N/A'):.3f}")
            report_lines.append(f"Reason: {explanation.get('reason', 'N/A')}")
            
            # Top contributing features
            contributions = explanation.get('feature_contributions', {})
            if contributions:
                report_lines.append("\nTop Contributing Features:")
                for i, (feature, value) in enumerate(list(contributions.items())[:5]):
                    report_lines.append(f"  {i+1}. {feature}: {value:.3f}")
            
            report_lines.append("")
        
        if len(flagged_indices) > 20:
            report_lines.append(f"\n... and {len(flagged_indices) - 20} more records")
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    from .detectors import IsolationForestDetector
    
    # Test explanation
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Smith", "Johnson", "RandomXYZ", "Williams"],
        "first_name": ["Hans", "John", "Jane", "Test123", "Bob"],
        "address": ["Main St 1", "Oak Ave 5", "Pine Rd 10", "123 Fake", "Elm St 20"],
        "email": ["hans.mueller@gmail.com", "john@yahoo.com", "jane@gmail.com", 
                 "x1y2z3@fake.net", "bob.w@outlook.com"],
        "iban": ["DE89370400440532013000", "GB82WEST12345698765432", 
                "DE11111111111111111111", "XX00000000000000000000", "FR7612345678901234567890"],
        "date_of_birth": ["1990-01-01", "1985-06-15", "1992-03-20", "2000-01-01", "1988-12-25"],
        "nationality": ["German", "British", "German", "Unknown", "French"],
    })
    
    detector = IsolationForestDetector(contamination=0.2, random_state=42)
    detector.fit(test_data)
    
    explainer = FraudExplainer(detector)
    
    print("Explanation for record 3:")
    explanation = explainer.explain_isolation_forest(test_data, 3)
    print(explanation)
    
    print("\nGlobal feature importance:")
    importance = explainer.get_global_feature_importance(test_data)
    print(importance)
