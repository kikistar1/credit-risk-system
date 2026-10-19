# ========== 算法扩展：新建LightGBM模型文件 ==========
"""
LightGBM 模型封装
算法工程师新增：实现和XGBoost一致的接口，方便接入训练流程和模型融合
"""
# ========== 算法扩展结束 ==========
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.lgb")

class LGBModel(BaseModel):
    def __init__(self, config, name: str = "lightgbm"):
        super().__init__(config, name)
        self.params = dict(config.params)
        self.model: Optional[LGBMClassifier] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        logger.info(f"开始训练 LightGBM - 样本: {len(X_train)}, 特征: {X_train.shape[1]}")
        if "scale_pos_weight" not in self.params:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        self.model = LGBMClassifier(**self.params, verbose=-1)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set)
        self._is_fitted = True
        self._feature_names = X_train.columns.tolist()

        if X_val is not None:
            from sklearn.metrics import roc_auc_score
            val_pred = self.model.predict_proba(X_val)[:, 1]
            logger.info(f"LightGBM 训练完成 - 验证集 AUC: {roc_auc_score(y_val, val_pred):.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self._feature_names:
            X = X[self._feature_names]
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        self._check_fitted()
        importance = pd.Series(self.model.feature_importances_, index=self._feature_names)
        return importance.sort_values(ascending=False)

    def save(self, file_path: str):
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        feature_path = path.with_suffix(".features.txt")
        feature_path.write_text("\n".join(self._feature_names), encoding="utf-8")
        logger.info(f"LightGBM 模型已保存至 {path}")

    def load(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        from lightgbm import Booster
        self.model = LGBMClassifier()
        self.model._Booster = Booster(model_file=str(path))
        feature_path = path.with_suffix(".features.txt")
        if feature_path.exists():
            self._feature_names = feature_path.read_text(encoding="utf-8").splitlines()
        self._is_fitted = True
        logger.info(f"LightGBM 模型已从 {path} 加载")
        return self
