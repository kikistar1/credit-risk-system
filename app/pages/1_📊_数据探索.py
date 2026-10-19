"""
📊 数据探索页面
================
EDA：数据概览、缺失值分析、目标分布、特征分布、相关性。
"""
import sys
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import distribution_plot, correlation_heatmap

st.set_page_config(page_title="数据探索", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 数据探索")
st.caption("理解你的数据是建模的第一步")


# ===== 数据加载 =====
from app.components.data_utils import find_data_file


@st.cache_data
def load_data(path_str: str, nrows: int | None = None) -> pd.DataFrame:
    """加载数据。path_str 必须是字符串以便 cache_data 哈希。"""
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


# 探测可用的数据源
real_path = ROOT / "data" / "raw" / "train.csv"
sample_path = ROOT / "data" / "sample" / "sample.csv"
real_available = real_path.exists()
sample_available = sample_path.exists()

if not real_available and not sample_available:
    st.error("❌ 未找到任何数据")
    st.info(
        "**方式 1**：从天池下载真实数据，放到 `data/raw/train.csv`\n\n"
        "**方式 2**：运行 `python scripts/generate_sample_data.py --n_samples 50000` 生成合成数据"
    )
    st.stop()

# ===== 数据源选择 =====
options = []
if real_available:
    options.append(f"📦 真实数据 (train.csv)")
if sample_available:
    options.append(f"🧪 合成数据 (sample.csv)")

col1, col2 = st.columns([3, 1])
with col1:
    data_source = st.radio(
        "选择数据源",
        options,
        index=0,  # 默认第一个：真实数据优先
        horizontal=True,
    )
with col2:
    if st.button("🔄 重新加载", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 真实数据 80 万行太大，可视化时只读前 N 行
if "真实" in data_source:
    target_path = real_path
    nrows = st.slider(
        "读取行数（真实数据较大，建议先用部分数据探索）",
        min_value=10_000, max_value=800_000, value=100_000, step=10_000,
    )
else:
    target_path = sample_path
    nrows = None

df = load_data(str(target_path), nrows=nrows)

if df.empty:
    st.error("⚠️ 数据加载失败")
    st.stop()

st.success(f"✅ 已加载 {len(df):,} 条数据，{df.shape[1]} 个字段")

# ===== Tab 1: 数据概览 =====
tab1, tab2, tab3, tab4 = st.tabs(["📋 数据概览", "🕳️ 缺失值分析", "🎯 目标分布", "📈 特征分析"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总样本数", f"{len(df):,}")
    col2.metric("特征数", df.shape[1])
    col3.metric("内存占用", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    if "isDefault" in df.columns:
        col4.metric("违约率", f"{df['isDefault'].mean():.2%}")

    st.markdown("### 数据样本（前 10 行）")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 字段类型分布")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["类型", "数量"]
    st.dataframe(dtype_counts, use_container_width=True, hide_index=True)

    with st.expander("📊 数值特征统计"):
        st.dataframe(df.describe().T, use_container_width=True)

    st.divider()
    st.markdown("### 🤖 LLM 智能数据洞察 (DeepSeek)")
    if st.button("✨ 一键生成数据诊断报告", type="primary"):
        from src.llm.client import DeepSeekClient
        from src.utils.config_loader import ConfigLoader
        
        with st.spinner("AI 风控专家正在阅读数据分布..."):
            try:
                configs = st.session_state.get("configs", ConfigLoader("./configs"))
                client = DeepSeekClient(configs.get("llm"))
                
                # 提取核心指标给 LLM
                missing_top = df.isnull().sum().sort_values(ascending=False).index[0]
                prompt = f"""
                作为一个资深金融风控数据科学家，请分析以下数据集情况并给出一段 150 字左右的数据洞察报告：
                1. 样本总数：{len(df):,} 条
                2. 包含特征：{df.shape[1]} 个
                3. 类别不平衡：违约率为 {df['isDefault'].mean():.2%}，存在明显的正负样本失衡。
                4. 缺失值痛点：缺失最严重的字段是 {missing_top}。
                
                请明确指出：这些数据痛点会对后续的 XGBoost 和 MLP 模型训练造成什么影响，并给出 2 条建模建议。
                """
                
                # 流式输出（等 C 同学把 client 封装好后，这里可以直接用 stream_chat）
                response = client.chat(prompt=prompt, system_prompt="你是一个严谨的数据科学家。")
                st.success(response)
            except Exception as e:
                st.error(f"LLM 调用失败，请确认 .env 已配置 Key。报错信息: {e}")
with tab2:
    st.markdown("### 缺失值统计")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("🎉 数据无缺失！")
    else:
        missing_df = pd.DataFrame({
            "字段": missing.index,
            "缺失数": missing.values,
            "缺失率": (missing.values / len(df) * 100).round(2),
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # 可视化
        import plotly.express as px
        fig = px.bar(
            missing_df,
            x="字段",
            y="缺失率",
            title="各字段缺失率（%）",
            color="缺失率",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 👇 你的 KNN 扩展点演示开始
        st.divider()
        st.markdown("### 🔬 扩展点演示：KNN 缺失值智能预测")
        st.info("💡 提示：直接填全局中位数会破坏数据分布。我们在底层只针对 `dti` 启用了 KNN，寻找相似邻居填补。")
        
        if st.button("🚀 现场预测缺失的 dti 值并生成对比图", type="primary"):
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import StandardScaler
            import plotly.graph_objects as go
            
            with st.spinner("正在调用底层 KNN 模型进行计算..."):
                try:
                    # 1. 拿 50 个真实用户
                    sample_users = df.dropna(subset=["annualIncome", "loanAmnt", "dti"]).head(50).copy()
                    original_dti = sample_users["dti"].values
                    global_median = df["dti"].median()
                    
                    # 2. 提取特征矩阵（只用年收入和贷款额算距离）
                    matrix = sample_users[["dti", "annualIncome", "loanAmnt"]].copy()
                    matrix["dti"] = np.nan # 挖空
                    
                    # 3. 标准化
                    scaler = StandardScaler()
                    matrix_scaled = scaler.fit_transform(matrix)
                    
                    # 4. 跑 KNN（找 5 个邻居）
                    imputer = KNNImputer(n_neighbors=5)
                    filled_matrix = imputer.fit_transform(matrix_scaled)
                    
                    # 5. 反标准化还原真实数值
                    knn_filled = filled_matrix[:, 0].reshape(-1, 1)
                    dti_scaler = StandardScaler()
                    dti_scaler.fit(original_dti.reshape(-1, 1))
                    knn_filled_raw = dti_scaler.inverse_transform(knn_filled).flatten()
                    
                    st.success("✅ KNN 扩展执行成功！下方是可视化对比结果：")

                    # 画图 1：柱状图
                    plot_df = pd.DataFrame({
                        "用户": [f"用户{i+1}" for i in range(50)],
                        "🔴 传统中位数": [global_median] * 50,
                        "🟢 KNN智能填充": knn_filled_raw,
                        "🔵 真实DTI": original_dti
                    })
                    
                    fig_bar = px.bar(
                        plot_df.head(15),
                        x="用户", y=["🔴 传统中位数", "🟢 KNN智能填充", "🔵 真实DTI"],
                        barmode="group",
                        title="缺失值填补效果对比（前15个用户）",
                        color_discrete_sequence=["#dc3545", "#28a745", "#0066cc"]
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # 画图 2：散点图
                    fig_scatter = px.scatter(
                        x=original_dti, y=knn_filled_raw,
                        labels={'x': '真实 DTI', 'y': 'KNN 填补的 DTI'},
                        title="KNN 填补准确度分布",
                        color_discrete_sequence=["#28a745"]
                    )
                    fig_scatter.add_trace(go.Scatter(
                        x=[min(original_dti), max(original_dti)],
                        y=[min(original_dti), max(original_dti)],
                        mode="lines", name="完美预测线", line=dict(color="gray", dash="dash")
                    ))
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)

                except Exception as e:
                    st.error(f"演示失败: {e}")
with tab3:
    if "isDefault" not in df.columns:
        st.warning("数据中没有 isDefault 字段，无法分析目标分布")
    else:
        col1, col2 = st.columns(2)
        with col1:
            target_dist = df["isDefault"].value_counts().reset_index()
            target_dist.columns = ["isDefault", "数量"]
            target_dist["isDefault"] = target_dist["isDefault"].map({0: "正常", 1: "违约"})

            fig = px.pie(
                target_dist,
                values="数量",
                names="isDefault",
                title="目标变量分布",
                color_discrete_sequence=["#28a745", "#dc3545"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 类别不平衡分析")
            n_pos = (df["isDefault"] == 1).sum()
            n_neg = (df["isDefault"] == 0).sum()

            st.metric("正常样本数", f"{n_neg:,}")
            st.metric("违约样本数", f"{n_pos:,}")
            st.metric("不平衡比例", f"1 : {n_neg/n_pos:.2f}")

            if n_neg / n_pos > 3:
                st.warning(
                    "⚠️ 数据存在类别不平衡，建议在模型训练时启用：\n"
                    "- XGBoost: `scale_pos_weight`\n"
                    "- MLP: 加权损失函数 / Focal Loss"
                )

with tab4:
    st.markdown("### 单特征分布")

    col1, col2 = st.columns([1, 3])
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "isDefault" in numeric_cols:
            numeric_cols.remove("isDefault")
        if "id" in numeric_cols:
            numeric_cols.remove("id")

        selected_col = st.selectbox("选择特征", numeric_cols)

    with col2:
        if selected_col:
            fig = distribution_plot(df, selected_col, target_col="isDefault" if "isDefault" in df.columns else None)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 特征相关性")
    if st.button("📊 生成相关性矩阵", help="对前 15 个数值特征计算相关性"):
        fig = correlation_heatmap(df, top_n=15)
        st.plotly_chart(fig, use_container_width=True)