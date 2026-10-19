import numpy as np
import pandas as pd
import shap
from dataclasses import dataclass
from typing import List, Dict

# --- 你之前写的占位类（现在和真正的分析器合并到一起） ---
@dataclass
class SHAPExplanation:
    base_value: float
    shap_values: np.ndarray
    feature_values: np.ndarray
    feature_names: List[str]
    prediction: float

    def get_top_features(self, top_n: int = 5) -> List[Dict]:
        abs_shap = np.abs(self.shap_values)
        top_idx = np.argsort(-abs_shap)[:top_n]
        return [
            {
                "feature": self.feature_names[i],
                "value": float(self.feature_values[i]) if isinstance(self.feature_values[i], (int, float, np.number)) else str(self.feature_values[i]),
                "shap_value": float(self.shap_values[i]),
                "direction": "↑ 增加风险" if self.shap_values[i] > 0 else "↓ 降低风险",
            }
            for i in top_idx
        ]

# --- A 同学原本应该有的 SHAP 分析器 ---
class SHAPAnalyzer:
    def __init__(self, max_background_size: int = 1000):
        self.explainer = None
        self.expected_value = 0.0
        self.feature_names = []
        self.max_background_size = max_background_size

    def fit(self, model, background_data):
        """构建 explainer"""
        if len(background_data) > self.max_background_size:
            background_data = background_data.sample(n=self.max_background_size, random_state=42)
        self.feature_names = background_data.columns.tolist()
        
        # 使用 TreeExplainer 加速
        self.explainer = shap.TreeExplainer(model)
        
        try:
            ev = self.explainer.expected_value
            self.expected_value = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        except Exception:
            self.expected_value = 0.5
            
        return self

    def explain_single(self, sample):
        """解释单个样本"""
        if isinstance(sample, pd.Series):
            sample_df = sample.to_frame().T
        else:
            sample_df = sample

        shap_values = self.explainer.shap_values(sample_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap_array = shap_values[0] if shap_values.ndim == 2 else shap_values
        prediction = self.expected_value + shap_array.sum()

        return SHAPExplanation(
            base_value=float(self.expected_value),
            shap_values=shap_array,
            feature_values=sample_df.iloc[0].values,
            feature_names=self.feature_names,
            prediction=float(prediction),
        )
    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """批量计算 SHAP 值"""
        if self.explainer is None:
            raise RuntimeError("请先调用 fit()")

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        return shap_values
    def global_importance(self, X, top_n: int = 20):
        """全局重要性"""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        importance = np.abs(shap_values).mean(axis=0)
        # 排序并截取 Top N
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False).head(top_n)