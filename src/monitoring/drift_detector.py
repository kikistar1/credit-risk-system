from scipy.stats import ks_2samp
import pandas as pd

class DriftDetector:
    def __init__(self, baseline_data: pd.DataFrame):
        self.baseline = baseline_data

    def detect(self, new_data: pd.DataFrame, threshold=0.1):
        """对每个特征做 KS 检验"""
        drifts = {}
        for col in self.baseline.columns:
            if col not in new_data.columns:
                continue
            
            # 仅对数值型特征做 KS 检验
            if not pd.api.types.is_numeric_dtype(self.baseline[col]):
                continue

            stat, pvalue = ks_2samp(
                self.baseline[col].dropna(),
                new_data[col].dropna(),
            )
            if stat > threshold:
                drifts[col] = {"ks_stat": stat, "p_value": pvalue}
        return drifts