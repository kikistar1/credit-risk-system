"""
评估指标模块
============
风控场景常用的评估指标：AUC、KS、PSI、混淆矩阵等。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

@dataclass
class EvalResult:
    """评估结果数据类"""
    auc: float = 0.0
    ks: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    log_loss: float = 0.0
    threshold: float = 0.5
    psi: Optional[float] = None  # 新增：指导书要求的扩展点

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"EvalResult(AUC={self.auc:.4f}, KS={self.ks:.4f}, "
            f"F1={self.f1:.4f} @ threshold={self.threshold:.2f})"
        )


class Evaluator:
    """模型评估器"""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = None,
        y_proba_baseline: np.ndarray = None,  # 指导书要求的入参
    ) -> EvalResult:
        """
        计算所有指标。
        
        指导书要求：如果传了 y_proba_baseline，则计算 PSI。
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        threshold = threshold if threshold is not None else self.threshold
        y_pred_label = (y_pred_proba >= threshold).astype(int)

        result = EvalResult(
            auc=roc_auc_score(y_true, y_pred_proba),
            ks=self.calculate_ks(y_true, y_pred_proba),
            accuracy=accuracy_score(y_true, y_pred_label),
            precision=precision_score(y_true, y_pred_label, zero_division=0),
            recall=recall_score(y_true, y_pred_label, zero_division=0),
            f1=f1_score(y_true, y_pred_label, zero_division=0),
            log_loss=log_loss(y_true, np.clip(y_pred_proba, 1e-15, 1 - 1e-15)),
            threshold=threshold,
        )
        
        # 完全遵照指导书 4.1.4 的扩展逻辑
        if y_proba_baseline is not None:
            result.psi = self._calculate_psi(y_pred_proba, y_proba_baseline)
            
        return result

    @staticmethod
    def calculate_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return float(np.max(np.abs(tpr - fpr)))

    @staticmethod
    def _calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        PSI (Population Stability Index) 计算核心逻辑。
        注意：按照 sklearn 习惯，expected 通常是基准分布（如训练集），actual 是新分布（如测试集）。
        """
        # 设定分箱边界
        breakpoints = np.arange(0, bins + 1) / (bins / 10)
        
        # 计算占比
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        # 防止除零
        expected_percents = np.clip(expected_percents, 0.0001, None)
        actual_percents = np.clip(actual_percents, 0.0001, None)

        # PSI 公式
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    @staticmethod
    def find_best_threshold(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = "youden",
    ) -> float:
        if method == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden = tpr - fpr
            return float(thresholds[np.argmax(youden)])
        elif method == "f1":
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
            return float(thresholds[np.argmax(f1_scores[:-1])])
        else:
            raise ValueError(f"不支持的方法: {method}")

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        y_pred = (y_pred_proba >= threshold).astype(int)
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return roc_curve(y_true, y_pred_proba)

    @staticmethod
    def get_pr_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return precision_recall_curve(y_true, y_pred_proba)

    @staticmethod
    def calculate_lift(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        sorted_indices = np.argsort(-y_pred_proba)
        y_sorted = y_true[sorted_indices]
        baseline = y_true.mean()
        bin_size = len(y_true) // n_bins
        lifts = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
            bin_rate = y_sorted[start:end].mean()
            lift = bin_rate / baseline if baseline > 0 else 0
            lifts.append({
                "bin": i + 1,
                "size": end - start,
                "default_rate": float(bin_rate),
                "lift": float(lift),
            })
        return {"baseline": float(baseline), "bins": lifts}