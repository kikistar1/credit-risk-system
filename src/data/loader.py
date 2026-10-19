"""数据加载器"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("data.loader")


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.paths = config.paths

    def load_train(self, use_sample: bool = False) -> pd.DataFrame:
        if use_sample:
            file_path = Path(self.paths.sample_dir) / self.paths.sample_file
        else:
            file_path = Path(self.paths.raw_dir) / self.paths.train_file
        return self._load_csv(file_path)

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {file_path}\n请运行 `python scripts/generate_sample_data.py` 生成数据。")
        loading_cfg = self.config.loading
        logger.info(f"加载数据: {file_path}")
        df = pd.read_csv(file_path, encoding=loading_cfg.encoding,
                         low_memory=loading_cfg.low_memory)
        logger.info(f"数据加载完成 - shape: {df.shape}")
        return df
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """打印数据基本信息摘要"""
        target = self.config.target_column
        summary = {
            "行数": df.shape[0],
            "列数": df.shape[1],
        }
        if target in df.columns:
            summary["违约人数"] = int(df[target].sum())
            summary["违约率"] = f"{df[target].mean():.2%}"
        logger.info(f"数据摘要: {summary}")
        return summary



class DataSplitter:
    def __init__(self, config):
        self.config = config
        self.split_cfg = config.split
        self.target = config.target_column
        self.id_col = config.id_column

    def split(self, df: pd.DataFrame, with_validation: bool = True) -> Tuple:
        drop_cols = [self.target]
        if self.id_col in df.columns:
            drop_cols.append(self.id_col)
        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target]
        stratify = y if self.split_cfg.stratify else None

        # 第一次划分：训练 vs 测试
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.split_cfg.test_size, random_state=self.split_cfg.random_state, stratify=stratify
        )
        if not with_validation:
            return X_trainval, X_test, y_trainval, y_test

        # 第二次划分：训练 vs 验证
        val_ratio = self.split_cfg.validation_size / \
            (1 - self.split_cfg.test_size)
        stratify2 = y_trainval if self.split_cfg.stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=self.split_cfg.random_state, stratify=stratify2
        )
        logger.info(
            f"数据划分完成 - 训练: {X_train.shape[0]}, 验证: {X_val.shape[0]}, 测试: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
