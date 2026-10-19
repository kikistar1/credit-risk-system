"""
Prompt 管理器
=============
集中管理所有 Prompt 模板，使用 Jinja2 实现变量插值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
from jinja2 import Environment, FileSystemLoader, Template

from src.utils.logger import get_logger

logger = get_logger("llm.prompts")


class PromptManager:
    def __init__(self, config):
        self.config = config
        self.templates_dir = Path(config.prompts.templates_dir)
        self.prompts_cfg = config.prompts

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Prompt 模板目录不存在: {self.templates_dir}")

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template_cache: Dict[str, Template] = {}

    def render(self, template_name: str, **kwargs) -> str:
        cfg = getattr(self.prompts_cfg, template_name, None)
        if cfg is None:
            raise KeyError(f"未配置的 Prompt 模板: {template_name}")

        template_file = cfg.template_file
        template = self._get_template(template_file)

        try:
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"渲染模板 {template_name} 失败: {e}")
            raise

    def get_system_role(self, template_name: str) -> str:
        cfg = getattr(self.prompts_cfg, template_name, None)
        if cfg is None:
            return ""
        return cfg.get("system_role", "")

    def _get_template(self, template_file: str) -> Template:
        if template_file not in self._template_cache:
            self._template_cache[template_file] = self.env.get_template(template_file)
        return self._template_cache[template_file]

    def reload(self) -> None:
        self._template_cache.clear()