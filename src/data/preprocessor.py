"""
数据预处理器
============
"""
from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from src.utils.logger import get_logger
logger = get_logger("data.preprocessor")


class Preprocessor:
    def __init__(self, data_config, feature_config):
        self.data_cfg = data_config
        self.feature_cfg = feature_config
        self._numerical_imputers: Dict[str, float] = {}
        self._categorical_imputers: Dict[str, str] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler = None
        self._dropped_cols: list = []
        self._outlier_clips: Dict[str, tuple] = {}

        # 新增两个变量，记录要填的列和用来算距离的列
        self._knn_fill_cols: List[str] = []
        self._knn_feature_cols: List[str] = []
        self._knn_imputer = None
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        logger.info(f"开始拟合预处理器 - 输入 shape: {df.shape}")
        df = df.copy()
        self._fit_drop_high_missing(df)
        df = df.drop(columns=self._dropped_cols, errors="ignore")
        self._fit_imputers(df)
        self._fit_encoders(df)
        self._fit_outlier_clips(df)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("预处理器未拟合，请先调用 fit() 或 fit_transform()")
        df = df.copy()
        df = df.drop(columns=self._dropped_cols, errors="ignore")
        df = self._apply_imputers(df)
        df = self._apply_outlier_clips(df)
        df = self._apply_encoders(df)
        df = self._handle_datetime(df)
        df = self._apply_scaler(df)
        return df

    def _fit_drop_high_missing(self, df: pd.DataFrame) -> None:
        threshold = self.feature_cfg.missing_value.drop_threshold
        missing_ratio = df.isnull().mean()
        self._dropped_cols = missing_ratio[missing_ratio > threshold].index.tolist()

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """学习缺失值填充策略【扩展KNN填充，支持yaml配置】"""
        num_strategy = self.feature_cfg.missing_value.numerical_strategy
        cat_strategy = self.feature_cfg.missing_value.categorical_strategy
        dict_strategy = self.feature_cfg.missing_value.strategy
        num_cols = [c for c in self.data_cfg.columns.numerical if c in df.columns]
        num_cols += [c for c in self.data_cfg.columns.anonymous if c in df.columns]

        self._knn_fill_cols.clear()
        # 遍历数值列，读取字段单独策略
        for c in num_cols:
            col_strategy = dict_strategy.get(c, num_strategy)
            if col_strategy == "knn":
                # 类型校验：KNN只能用于数值列
                if not pd.api.types.is_numeric_dtype(df[c]):
                    raise ValueError(f"字段 {c} 不是数值类型，不能配置为knn填充")
                self._knn_fill_cols.append(c)

        self._knn_feature_cols = num_cols  # 用来算距离的所有数值列
        # 如果存在 knn 列，才启动 KNN 模型
        if len(self._knn_fill_cols) > 0:
            # 从yaml读取KNN近邻参数，兜底默认5
            knn_cfg = self.feature_cfg.missing_value.get("knn", {})
            n_neighbors = knn_cfg.get("n_neighbors", 5)
            self._knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            self._knn_imputer.fit(df[self._knn_feature_cols])
            logger.info(f"KNN拟合完成，待填补列：{self._knn_fill_cols}，k={n_neighbors}")

        # 其他没配 knn 的列，读取字段单独策略填充
        for col in num_cols:
            if col in self._knn_fill_cols:
                continue
            col_strategy = dict_strategy.get(col, num_strategy)
            if col_strategy == "mean":
                self._numerical_imputers[col] = df[col].mean()
            elif col_strategy == "median":
                self._numerical_imputers[col] = df[col].median()
            elif col_strategy == "zero":
                self._numerical_imputers[col] = 0
            else:
                self._numerical_imputers[col] = df[col].median()

        # 类别列
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        for col in cat_cols:
            if cat_strategy == "mode":
                mode = df[col].mode()
                self._categorical_imputers[col] = mode.iloc[0] if len(mode) > 0 else "unknown"
            else:
                self._categorical_imputers[col] = "unknown"

    def _apply_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用缺失值填充"""
        # 1. 执行 KNN 填充（精准狙击：只覆盖配置为KNN的列）
        if self._knn_imputer is not None and len(self._knn_fill_cols) > 0:
            # 获取所有列的填补结果
            imputed_array = self._knn_imputer.transform(df[self._knn_feature_cols])
            imputed_df = pd.DataFrame(imputed_array, columns=self._knn_feature_cols, index=df.index)

            # 只覆盖原来配置了 knn 的列（比如只覆盖 dti）
            for col in self._knn_fill_cols:
                df[col] = imputed_df[col]
        # 2. 其他列走中位数/均值填充
        for col, fill_value in self._numerical_imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        # 3. 类别列填充
        for col, fill_value in self._categorical_imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value).astype(str)
        return df

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        custom_mappings = self.feature_cfg.encoding.get("custom", {})
        for col in cat_cols:
            if col in custom_mappings:
                continue
            le = LabelEncoder()
            values = df[col].fillna("unknown").astype(str)
            le.fit(values)
            self._label_encoders[col] = le

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        custom_mappings = self.feature_cfg.encoding.get("custom", {})
        for col, cfg in custom_mappings.items():
            if col in df.columns and "mapping" in cfg:
                df[col] = df[col].astype(str).map(cfg["mapping"]).fillna(-1).astype(int)
        for col, le in self._label_encoders.items():
            if col in df.columns:
                values = df[col].fillna("unknown").astype(str)
                known = set(le.classes_)
                values = values.map(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(values)
        return df

    def _fit_outlier_clips(self, df: pd.DataFrame) -> None:
        outlier_cfg = self.feature_cfg.outlier
        if outlier_cfg.method != "clip":
            return
        for col in outlier_cfg.apply_to:
            if col in df.columns:
                self._outlier_clips[col] = (df[col].quantile(outlier_cfg.lower_quantile),
                                           df[col].quantile(outlier_cfg.upper_quantile))

    def _apply_outlier_clips(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self._outlier_clips.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _handle_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.data_cfg.columns.datetime:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
                df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
                df = df.drop(columns=[col])
            except Exception as e:
                df = df.drop(columns=[col], errors="ignore")
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.feature_cfg.scaling.method
        if method == "none":
            return df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        if self._scaler is None:
            if method == "standard":
                self._scaler = StandardScaler()
            elif method == "minmax":
                self._scaler = MinMaxScaler()
            elif method == "robust":
                self._scaler = RobustScaler()
            else:
                return df
            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        else:
            scaler_cols = [c for c in numeric_cols if hasattr(self._scaler, "mean_")]
            if scaler_cols:
                df[scaler_cols] = self._scaler.transform(df[scaler_cols])
        return df
