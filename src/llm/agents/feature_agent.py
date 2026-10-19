import json
import re
import pandas as pd
from typing import Any, Dict
from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.feature_agent")

FIELD_DESCRIPTIONS = {
    "loanAmnt": "贷款金额（元）", "term": "贷款期限（年）", "interestRate": "贷款利率（%）",
    "installment": "分期付款金额（元）", "grade": "贷款等级（A-G）", "subGrade": "贷款等级子级",
    "employmentTitle": "就业职称（脱敏）", "employmentLength": "就业年限", "homeOwnership": "房屋所有权状况",
    "annualIncome": "年收入（元）", "verificationStatus": "验证状态", "issueDate": "贷款发放日期",
    "purpose": "贷款用途", "dti": "负债收入比（%）", "delinquency_2years": "近 2 年逾期次数",
    "ficoRangeLow": "FICO 评分下限", "ficoRangeHigh": "FICO 评分上限", "openAcc": "未结信用额度数",
    "pubRec": "公开记录数", "pubRecBankruptcies": "破产记录数", "revolBal": "循环余额（元）",
    "revolUtil": "循环额度利用率（%）", "totalAcc": "总信用额度数"
}

class FeatureAgent:
    def __init__(self, client: DeepSeekClient, prompt_manager: PromptManager, config):
        self.client = client
        self.prompts = prompt_manager
        self.config = config

    def suggest(self, df: pd.DataFrame, current_auc: float = 0.7, current_model: str = "XGBoost", bottleneck: str = None) -> Dict[str, Any]:
        logger.info(f"生成特征建议 - 数据规模: {df.shape}")
        context = self._build_context(df, current_auc, current_model, bottleneck)
        prompt = self.prompts.render("feature_suggestion", **context)
        system_prompt = self.prompts.get_system_role("feature_suggestion")
        response = self.client.chat(prompt, system_prompt=system_prompt)
        return self._parse_response(response)

    def _build_context(self, df, current_auc, current_model, bottleneck) -> Dict[str, Any]:
        sample_size = min(self.config.sample_size, len(df))
        fields = []
        for col in df.columns:
            field_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "description": FIELD_DESCRIPTIONS.get(col, "（未知字段）"),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                field_info["stats"] = f"均值={stats['mean']:.2f}, 中位数={stats['50%']:.2f}"
            fields.append(field_info)

        data_sample = df.head(5).to_string(max_cols=10)
        return {
            "sample_size": sample_size,
            "feature_count": df.shape[1],
            "fields": fields,
            "data_sample": data_sample,
            "current_model": current_model,
            "current_auc": current_auc,
            "bottleneck": bottleneck,
            "max_suggestions": self.config.max_suggestions,
        }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {"suggestions": [], "general_advice": response}