"""特征构造器"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("features.builder")

class FeatureBuilder:
    def __init__(self, config):
        self.config = config
        self.derived_cfg = config.derived_features
        self._created_features = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self.derived_cfg, "enabled", False): return df
        df = df.copy()
        for rule in self.derived_cfg.rules:
            try:
                name = rule["name"]
                formula = rule["formula"]
                if name not in df.columns:
                    safe_globals = {"df": df, "pd": pd, "np": np, "__builtins__": {}}
                    df[name] = eval(formula, safe_globals)
                self._created_features.append(name)
            except Exception as e:
                logger.warning(f"创建特征失败: {e}")
        return df
    # 兼容：build 和 transform 指向同一个方法
    transform=build

    def get_created_features(self) -> List[str]:
        return self._created_features.copy()