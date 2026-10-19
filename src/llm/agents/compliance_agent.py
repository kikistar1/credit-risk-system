import json
import re
from src.llm.client import BaseLLMClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.compliance_agent")

class ComplianceAgent:
    DEFAULT_RULES = [
        "贷款利率不得超过 LPR 的 4 倍（约 14% 左右）",
        "对 60 岁以上申请人，需要额外的还款能力证明",
        "贷款用途禁止用于房地产投资",
    ]

    def __init__(self, client: BaseLLMClient, prompt_manager: PromptManager, config):
        self.client = client
        self.pm = prompt_manager
        self.config = config

    def check(self, features, decision, interest_rate, rules=None):
        rules_to_use = rules or self.DEFAULT_RULES
        
        # 1. 第一次常规审查
        prompt = self.pm.render(
            "compliance_check",
            features=features,
            decision=decision,
            interest_rate=interest_rate,
            rules=rules_to_use,
        )
        response = self.client.chat(
            prompt=prompt,
            system_prompt=self.pm.get_system_role("compliance_check"),
            temperature=0.1,
            max_tokens=1000,
        )
        first_result = self._parse_response(response)

        # 2. 反思（复核）机制（方向3）
        # 如果第一次认为合规，再让 AI 自我反问一下，防止幻觉漏判
        if first_result.get("is_compliant", True):
            reflect_prompt = f"""你是一个严苛的合规复核员。之前的审查员认为以下情况是合规的：
- 特征：{json.dumps(features, ensure_ascii=False)}
- 决策：{decision}
- 利率：{interest_rate}%
- 规则：{rules_to_use}

请谨慎检查是否存在**被遗漏**的风险点？
如果确实没有风险，请输出 JSON：{{"is_compliant": true, "reason": "复核确认无误"}}
如果发现了风险，请输出 JSON：{{"is_compliant": false, "reason": "复核发现新的风险点"}}
仅输出 JSON。"""
            try:
                reflect_resp = self.client.chat(reflect_prompt, temperature=0.1, max_tokens=200)
                reflect_json = self._parse_response(reflect_resp)
                # 如果复核员推翻了结论，则更新结果并补充违规项
                if not reflect_json.get("is_compliant", True):
                    first_result["is_compliant"] = False
                    first_result.setdefault("violations", []).append({
                        "rule": "复核发现潜在风险",
                        "reason": reflect_json.get("reason", "复核不通过")
                    })
            except Exception as e:
                logger.warning(f"复核阶段失败，跳过复核: {e}")

        return first_result

    def _parse_response(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text.replace("`", ""))
        except Exception:
            return {"raw_response": text}