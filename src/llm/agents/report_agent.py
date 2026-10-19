from typing import Any, Dict
from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.report_agent")

class ReportAgent:
    def __init__(self, client: DeepSeekClient, prompt_manager: PromptManager, config):
        self.client = client
        self.prompts = prompt_manager
        self.config = config
        self.top_n = config.top_n_features

    def generate(self, features: Dict[str, Any], probability: float, shap_explanation, stream: bool = False):
        context = self._build_context(features, probability, shap_explanation)
        prompt = self.prompts.render("risk_report", **context)
        system_prompt = self.prompts.get_system_role("risk_report")

        logger.info(f"生成风险报告 - 违约概率={probability:.4f}")

        if stream:
            return self.client.stream_chat(prompt, system_prompt=system_prompt)
        return self.client.chat(prompt, system_prompt=system_prompt)

    def _build_context(self, features, probability, shap_explanation) -> Dict[str, Any]:
        if probability < 0.3:
            risk_level = "低风险"
        elif probability < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        top_features = shap_explanation.get_top_features(self.top_n)
        display_features = self._select_display_features(features)

        return {
            "features": display_features,
            "probability": probability,
            "risk_level": risk_level,
            "shap_top_features": top_features,
            "top_n": self.top_n,
        }

    @staticmethod
    def _select_display_features(features: Dict) -> Dict:
        priority_fields = [
            "loanAmnt", "term", "interestRate", "installment",
            "annualIncome", "dti", "employmentLength", "homeOwnership",
            "purpose", "grade", "ficoRangeLow", "ficoRangeHigh",
            "delinquency_2years", "openAcc", "totalAcc", "revolUtil",
        ]
        labels = {
            "loanAmnt": "贷款金额", "term": "贷款期限(年)", "interestRate": "利率(%)",
            "installment": "月还款额", "annualIncome": "年收入", "dti": "负债收入比",
            "employmentLength": "就业年限", "homeOwnership": "房屋所有权", "purpose": "贷款用途",
            "grade": "贷款等级", "ficoRangeLow": "FICO 评分下限", "ficoRangeHigh": "FICO 评分上限",
            "delinquency_2years": "近2年逾期次数", "openAcc": "未结信用额度数",
            "totalAcc": "总信用额度数", "revolUtil": "循环额度利用率",
        }
        result = {}
        for field in priority_fields:
            if field in features:
                label = labels.get(field, field)
                value = features[field]
                if isinstance(value, float):
                    value = round(value, 2)
                result[label] = value
        return result