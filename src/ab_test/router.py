"""
A/B 测试路由模块
================
通过一致性哈希，将用户稳定分配到 A 组或 B 组。
"""
import hashlib
import pandas as pd
from typing import Tuple, Any

class ABTestRouter:
    def __init__(self, model_a: Any, model_b: Any, ratio: float = 0.5):
        """
        参数:
            model_a: A组模型（例如基线 XGBoost）
            model_b: B组模型（例如新模型 LightGBM 或融合模型）
            ratio: A组的分流比例，0.5 代表 50% A组，50% B组
        """
        self.model_a = model_a
        self.model_b = model_b
        self.ratio = ratio

    def route(self, user_id: str, features: pd.DataFrame) -> Tuple[str, float]:
        """
        路由分发并预测。
        参数:
            user_id: 用户唯一标识（如 id）
            features: 预处理后的特征 DataFrame
        返回:
            (分配到的组别 "A"/"B", 预测的违约概率)
        """
        # 一致性哈希：保证同一个 user_id 每次都会分到同一个组
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100.0
        
        if bucket < self.ratio:
            # 走 A 模型
            proba = self.model_a.predict_proba(features)[0]
            return "A", float(proba)
        else:
            # 走 B 模型
            proba = self.model_b.predict_proba(features)[0]
            return "B", float(proba)