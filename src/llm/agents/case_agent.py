import numpy as np
import pandas as pd
from typing import Any, Dict
from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.case_agent")

class CaseAgent:
    def __init__(self, client: DeepSeekClient, prompt_manager: PromptManager, config):
        self.client = client
        self.prompts = prompt_manager
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.max_cases = config.max_cases

    def find_anomalies(self, y_true: np.ndarray, y_pred_proba: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        wrong_mask = (y_pred != y_true)
        confidence = np.abs(y_pred_proba - 0.5) * 2
        high_confidence_mask = confidence >= self.confidence_threshold
        anomaly_mask = wrong_mask & high_confidence_mask

        if not anomaly_mask.any():
            logger.info("未发现高置信度的错误案例")
            return pd.DataFrame()

        error_types = np.where(
            (y_true == 0) & (y_pred == 1), "False Positive (误判违约)",
            np.where((y_true == 1) & (y_pred == 0), "False Negative (漏判违约)", "其他")
        )

        result = pd.DataFrame({
            "index": np.where(anomaly_mask)[0],
            "true_label": y_true[anomaly_mask],
            "pred_proba": y_pred_proba[anomaly_mask],
            "confidence": confidence[anomaly_mask],
            "error_type": error_types[anomaly_mask],
        })
        result = result.sort_values("confidence", ascending=False).head(self.max_cases)
        return result.reset_index(drop=True)

    def analyze(self, features: Dict[str, Any], true_label: int, probability: float, shap_explanation, stream: bool = False):
        context = self._build_context(features, true_label, probability, shap_explanation)
        prompt = self.prompts.render("case_analysis", **context)
        system_prompt = self.prompts.get_system_role("case_analysis")

        logger.info(f"分析异常案例 - 真实={true_label}, 预测概率={probability:.4f}")
        if stream:
            return self.client.stream_chat(prompt, system_prompt=system_prompt)
        return self.client.chat(prompt, system_prompt=system_prompt)

    def _build_context(self, features, true_label, probability, shap_explanation) -> Dict[str, Any]:
        true_label_text = "违约" if true_label == 1 else "正常"
        predicted_label = 1 if probability >= 0.5 else 0
        predicted_label_text = "违约" if predicted_label == 1 else "正常"
        confidence = abs(probability - 0.5) * 2
        confidence_level = f"非常高 ({confidence:.2%})" if confidence >= 0.85 else f"一般 ({confidence:.2%})"

        shap_features = shap_explanation.get_top_features(top_n=5)
        priority = ["loanAmnt", "term", "interestRate", "annualIncome", "dti", "ficoRangeLow", "revolUtil"]
        display_features = {k: features[k] for k in priority if k in features}

        return {
            "features": display_features,
            "true_label": true_label,
            "true_label_text": true_label_text,
            "predicted_label": predicted_label,
            "predicted_label_text": predicted_label_text,
            "probability": probability,
            "confidence_level": confidence_level,
            "shap_features": shap_features,
        }