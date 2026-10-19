from __future__ import annotations

import json
import os
import time
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI
import httpx
from src.utils.logger import get_logger

logger = get_logger("llm.client")

# ==========================================
# 扩展点：工厂基类与各厂商实现
# ==========================================
class BaseLLMClient:
    def chat(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict]] = None, **kwargs) -> str:
        raise NotImplementedError
    def stream_chat(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict]] = None, **kwargs) -> Generator[str, None, None]:
        raise NotImplementedError


class DeepSeekClient(BaseLLMClient):
    def __init__(self, config):
        self.config = config
        self.deepseek_cfg = config.deepseek
        
        api_key = os.environ.get(self.deepseek_cfg.api_key_env)
        self.client = OpenAI(
            api_key=api_key or "dummy-key",
            base_url="https://api.deepseek.com/v1",
            timeout=self.deepseek_cfg.default_params.timeout,
            http_client=httpx.Client(trust_env=False)
        )

        self.model = "deepseek-chat"
        self.default_params = dict(self.deepseek_cfg.default_params)
        self.default_params.pop("timeout", None)
        
        # 缓存与预算
        self.cache_dir = Path("./data/llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_file = Path("./data/llm_budget.json")
        self.daily_limit = 100000  # 每日 10 万 Token 预算上限

    def _check_budget(self, current_usage: int = 0):
        """扩展点：Token 预算控制"""
        today = str(date.today())
        budget_data = {"date": today, "used_tokens": 0}
        
        if self.budget_file.exists():
            try:
                budget_data = json.loads(self.budget_file.read_text(encoding="utf-8"))
                if budget_data.get("date") != today:
                    budget_data = {"date": today, "used_tokens": 0}
            except Exception:
                pass
        
        # 检查是否超限
        if budget_data["used_tokens"] + current_usage > self.daily_limit:
            raise RuntimeError(f"⛔ 每日 Token 预算超限 ({budget_data['used_tokens']}/{self.daily_limit})，请明日再试。")
        
        # 记录消耗
        if current_usage > 0:
            budget_data["used_tokens"] += current_usage
            self.budget_file.write_text(json.dumps(budget_data), encoding="utf-8")

    def _get_cache_key(self, prompt: str, system_prompt: str) -> str:
        content = f"{self.model}_{system_prompt}_{prompt}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def chat(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict]] = None, **kwargs) -> str:
        # 缓存判断
        if not history and "max_tokens" not in kwargs:
            cache_key = self._get_cache_key(prompt, system_prompt or "")
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                logger.info("命中本地缓存，直接返回！")
                return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

        self._check_budget(0) # 调用前先检查有没有超限
        
        messages = self._build_messages(prompt, system_prompt, history)
        params = {**self.default_params, **kwargs}
        
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=False, **params
        )
        duration = time.time() - t0
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else {}
        
        # 记录 token 消耗
        self._check_budget(usage.get("total_tokens", 0))
        
        # 写入缓存
        if not history and "max_tokens" not in kwargs:
            cache_file.write_text(json.dumps({"response": content}, ensure_ascii=False), encoding="utf-8")

        return content

    def stream_chat(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict]] = None, **kwargs) -> Generator[str, None, None]:
        self._check_budget(0)
        messages = self._build_messages(prompt, system_prompt, history)
        params = {**self.default_params, **kwargs}
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **params
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"流式调用失败: {e}")
            yield f"\n\n[错误]: {e}"

    def _build_messages(self, prompt, system_prompt, history):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages


class TongyiClient(DeepSeekClient):
    """扩展点：通义千问实现（如果以后要用，改一下 base_url 和 model 即可）"""
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", "dummy"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-plus"


# ==========================================
# 工厂模式分发器
# ==========================================
class LLMClient:
    @classmethod
    def from_config(cls, config):
        # 默认就是 deepseek，你可以在 yaml 里加一行 provider: tongyi 切换
        provider = getattr(config.deepseek, "provider", "deepseek").lower()
        
        if provider == "deepseek":
            return DeepSeekClient(config)
        elif provider == "tongyi":
            return TongyiClient(config)
        else:
            raise ValueError(f"未知的 provider: {provider}")