"""
📈 模型监控页面
================
监控数据漂移 (Data Drift)
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from src.monitoring.drift_detector import DriftDetector
from app.components.data_utils import find_data_file

st.set_page_config(page_title="模型监控", page_icon="📈", layout="wide")
render_sidebar()

st.title("📈 模型监控与数据漂移检测")
st.caption("监控线上新数据分布与训练数据分布的差异 (基于 KS 检验)")

# 1. 获取数据
data_path, source = find_data_file(ROOT, prefer="auto")
if data_path is None:
    st.error("未找到数据，请先生成或下载数据。")
    st.stop()

df = pd.read_csv(data_path)
target = "isDefault"
id_col = "id"
# 把标签和 ID 都剔除，不参与漂移检测
if target in df.columns:
    df = df.drop(columns=[target])
if id_col in df.columns:
    df = df.drop(columns=[id_col])
# 2. 模拟划分基准集（训练时）和 新数据集（线上）
# 实际生产中，基准集应该是训练集的存档，新数据是近期产生的
# 找到原来的这两行：
# train_df = df.head(int(len(df) * 0.5))
# new_df = df.tail(int(len(df) * 0.5))

# 【替换为以下代码】
train_df = df.head(int(len(df) * 0.5))
new_df = df.tail(int(len(df) * 0.5)).copy()

# 新增：加一个开关，用来在答辩现场演示“数据漂移”
simulate_drift = st.checkbox("⚠️ 模拟线上发生重大业务变更（勾选后，利率将整体上调 15%）")

if simulate_drift:
    # 假设利率列叫 interestRate，我们人为把它调高 15 个点，模拟恶性通胀或客群下沉
    if 'interestRate' in new_df.columns:
        new_df['interestRate'] = new_df['interestRate'] + 15
        st.warning("已人为篡改新数据集中的【利率】特征，预计监控系统将报警！")
st.info(f"当前使用 **{source}** 数据进行模拟监控。基准集样本数：{len(train_df)}，新数据集样本数：{len(new_df)}")

threshold = st.slider("KS 统计量阈值 (超过此值认为发生漂移)", 0.01, 0.5, 0.1, 0.01)

if st.button("🔬 开始漂移检测", type="primary"):
    with st.spinner("正在计算各特征 KS 统计量..."):
        detector = DriftDetector(baseline_data=train_df)
        drift_results = detector.detect(new_data=new_df, threshold=threshold)
        
    if not drift_results:
        st.success(f"✅ 监控正常：所有特征的 KS 统计量均未超过阈值 {threshold}，未发生明显漂移。")
    else:
        st.error(f"🚨 警告：发现 {len(drift_results)} 个特征发生了严重漂移 (KS > {threshold})！建议重新训练模型。")
        
        # 将结果转为 DataFrame 并展示
        result_df = pd.DataFrame.from_dict(drift_results, orient='index').reset_index()
        result_df.columns = ['特征名称', 'KS统计量', 'P值']
        result_df = result_df.sort_values(by='KS统计量', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### 漂移特征列表")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("### KS 统计量分布图")
            fig = px.bar(
                result_df, 
                x='特征名称', 
                y='KS统计量',
                color='KS统计量',
                color_continuous_scale='Reds',
                title="发生漂移的特征 (红线为阈值)"
            )
            fig.add_hline(y=threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)