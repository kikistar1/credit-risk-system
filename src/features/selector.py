"""特征筛选器"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2
from src.utils.logger import get_logger

logger = get_logger("features.selector")

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        self._selected_features: Optional[List[str]] = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        if not getattr(self.config, "enabled", True):
            self._selected_features = X.columns.tolist()
            return self
            
        current_cols = X.columns.tolist()
        for method_cfg in self.config.methods:
            method_name = method_cfg["name"]
            if method_name == "variance":
                selector = VarianceThreshold(threshold=method_cfg.get("threshold", 0.01))
                selector.fit(X[current_cols].fillna(0))
                current_cols = X[current_cols].columns[selector.get_support()].tolist()
            elif method_name == "correlation":
                threshold = method_cfg.get("threshold", 0.95)
                corr = X[current_cols].corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
                current_cols = [c for c in current_cols if c not in to_drop]
            elif method_name == "model_based":
                from xgboost import XGBClassifier
                # ==========================================
                # 【改动】加了n_jobs=-1，用满CPU多核
                # ==========================================
                model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                     n_jobs=-1, verbosity=0, eval_metric="logloss")
                model.fit(X[current_cols].fillna(0), y)
                importance = pd.Series(model.feature_importances_, index=current_cols)
                current_cols = importance.nlargest(method_cfg.get("top_k", 50)).index.tolist()
            elif method_name == "mutual_info":
                # 🚀 扩展：互信息
                # ==========================================
                # 【改动】加了n_jobs=-1，用满CPU多核
                # ==========================================
                mi_scores = mutual_info_classif(X[current_cols].fillna(0), y, random_state=42, n_jobs=-1)
                importance = pd.Series(mi_scores, index=current_cols)
                current_cols = importance.nlargest(method_cfg.get("top_k", 45)).index.tolist()
            # ========== 算法扩展：卡方检验特征选择 ==========
            elif method_name == "chi2":
                # 卡方检验特征选择：筛选和目标相关性最高的特征
                from sklearn.preprocessing import MinMaxScaler
                X_scaled = MinMaxScaler().fit_transform(X[current_cols].fillna(0))
                selector = SelectKBest(chi2, k=method_cfg.get("top_k", 40))
                selector.fit(X_scaled, y)
                current_cols = X[current_cols].columns[selector.get_support()].tolist()
            # ========== 算法扩展结束 ==========

        self._selected_features = current_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None: return X
        return X[[c for c in self._selected_features if c in X.columns]]

    @property
    def selected_features(self) -> List[str]:
        return self._selected_features or []