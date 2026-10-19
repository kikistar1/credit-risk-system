"""
💬 LLM 智能助手页面 (最终完美版)
====================
包含：自由对话、风险报告、特征建议、异常分析、合规检查、预算监控。
"""
import os
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# 【改动点 1】为了避免 D 同学没写侧边栏报错，这里依然保持注释状态
from app.components.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="LLM 智能助手", page_icon="💬", layout="wide")
st.title("💬 LLM 智能助手")
st.caption("基于 DeepSeek 大模型的风控领域 AI 助手")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))

# ===== 检查 API Key =====
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key.startswith("sk-") or api_key == "sk-your-api-key-here":
    st.error("⚠️ 未配置 DeepSeek API Key")
    st.stop()

# ==========================================
# 【改动点 2】初始化客户端：改成了工厂模式 LLMClient
# ==========================================
@st.cache_resource
def get_llm_client():
    # 之前是导入 DeepSeekClient，现在导入 LLMClient
    from src.llm.client import LLMClient
    from src.llm.prompts import PromptManager
    # 之前是 DeepSeekClient(configs.get("llm"))，现在是 LLMClient.from_config(...)
    return LLMClient.from_config(configs.get("llm")), PromptManager(configs.get("llm"))

client, prompt_mgr = get_llm_client()

# ==========================================
# 【改动点 3】模式选择里，去掉了单独的 radio，直接把所有模式合并在最上面的这一个 radio 里
# ==========================================
mode = st.radio(
    "选择对话模式",
    ["💬 自由对话", "📝 风险报告", "💡 特征建议", "🔬 异常案例分析", "⚖️ 政策合规检查", "🛡️ 预算与架构监控"], # 加了最后一个
    horizontal=True,
)
st.divider()

# ===== 模式：自由对话 =====
if mode == "💬 自由对话":
    st.markdown("##### 💡 你可以问我关于数据、模型、风控业务的任何问题")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:= st.chat_input("请输入您的问题..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            latest_dir = ROOT / "data" / "models" / "latest"
            model_info, dataset_info = {}, {}
            eval_path = latest_dir / "eval_results.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                model_info = {
                    "xgb_auc": round(eval_data.get("xgboost", {}).get("auc", 0), 4),
                    "mlp_auc": round(eval_data.get("mlp", {}).get("auc", 0), 4),
                    "ensemble_auc": round(eval_data.get("ensemble", {}).get("auc", 0), 4),
                }

            prompt = prompt_mgr.render(
                "chat_assistant",
                user_question=user_input,
                model_info=model_info if model_info else None,
                dataset_info=dataset_info if dataset_info else None,
            )
            system_prompt = prompt_mgr.get_system_role("chat_assistant")

            max_history = configs.get("llm").agents.chat_agent.max_history_turns
            history = [{"role": h["role"], "content": h["content"]} for h in st.session_state.chat_messages[-max_history * 2:-1]]

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in client.stream_chat(prompt, system_prompt=system_prompt, history=history):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"调用失败: {e}")

    if st.session_state.chat_messages:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_messages = []
            st.rerun()

# ===== 模式：风险报告 =====
elif mode == "📝 风险报告":
    st.info("ℹ️ 完整的风险报告功能在 **🎯 单笔预测** 页面，本处提供快速演示。")
    st.markdown("##### 输入贷款信息，让 LLM 生成风险评估报告")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("贷款金额", 1000, 500000, 30000)
        annual_income = st.number_input("年收入", 10000, 1000000, 80000)
    with col2:
        interest_rate = st.slider("利率(%)", 5.0, 30.0, 13.5)
        dti = st.slider("DTI(%)", 0.0, 60.0, 18.0)
    with col3:
        fico = st.slider("FICO 分数", 580, 850, 690)
        proba = min(max(0.05 + (interest_rate - 5) * 0.02 + dti * 0.005 - (fico - 580) * 0.001, 0.0), 0.95)
        st.metric("模拟违约概率", f"{proba:.2%}")

    if st.button("🤖 生成简化报告", type="primary"):
        prompt = f"""请基于以下信贷申请信息，生成一份简短的风险评估意见（300字以内）：
- 贷款金额: ¥{loan_amnt:,}，利率: {interest_rate}%，年收入: ¥{annual_income:,}
- 负债收入比: {dti}%，FICO 分数: {fico}，违约概率: {proba:.2%}
请给出: 1) 风险等级 2) 主要担忧点 3) 审批建议"""
        try:
            with st.spinner("生成中..."):
                placeholder = st.empty()
                full = ""
                for chunk in client.stream_chat(prompt):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
        except Exception as e:
            st.error(f"调用失败: {e}")

# ===== 模式：特征建议 =====
elif mode == "💡 特征建议":
    st.markdown("##### 让 AI 帮你提出特征工程建议")

    if st.button("🚀 生成建议", type="primary"):
        try:
            from src.llm.agents.feature_agent import FeatureAgent
            # 【改动点 4】去掉了对 D 同学 data_utils 的依赖，直接原生读文件，防止报错崩溃
            data_path = ROOT / "data" / "sample" / "sample.csv"
            if not data_path.exists():
                data_path = ROOT / "data" / "raw" / "train.csv"
            
            if not data_path.exists():
                st.error("❌ 未找到数据。请先在终端运行 `python scripts/generate_sample_data.py --n_samples 10000` 生成数据。")
                st.stop()

            df = pd.read_csv(data_path, nrows=1000)
            st.caption(f"使用数据：{data_path.name}（{len(df):,} 行）")

            agent = FeatureAgent(client, prompt_mgr, configs.get("llm").agents.feature_agent)

            with st.spinner("AI 思考中..."):
                result = agent.suggest(df=df, current_auc=0.72)

            suggestions = result.get("suggestions", [])
            if suggestions:
                for i, sug in enumerate(suggestions):
                    with st.expander(f"💡 建议 {i+1}: {sug.get('name', '未命名')}"):
                        st.markdown(f"**业务含义**: {sug.get('business_meaning', '')}")
                        st.markdown(f"**预期效果**: {sug.get('expected_impact', '')}")
                        st.code(sug.get("code", ""), language="python")

            if "general_advice" in result:
                st.info(f"**整体建议**: {result['general_advice']}")
        except Exception as e:
            st.error(f"调用失败: {e}")

# ===== 模式：异常分析 =====
elif mode == "🔬 异常案例分析":
    st.markdown("##### 分析模型预测错误的高置信度案例")
    st.markdown("**示例：分析一个 False Negative（漏判违约）**")

    if st.button("🔬 演示分析", type="primary"):
        prompt = """请分析以下机器学习模型预测错误的案例：
【真实情况】客户最终违约（label=1）
【模型预测】违约概率仅 5%（高置信度的错误判断）
【客户特征】贷款 45,000，年收入 120,000，FICO 750，DTI 12%，就业 8 年，自有房屋，无逾期。
请从数据、特征、模型、业务四个角度分析：为什么模型预测错误？并给出 3 条改进建议。"""
        try:
            with st.spinner("分析中..."):
                placeholder = st.empty()
                full = ""
                for chunk in client.stream_chat(prompt):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
        except Exception as e:
            st.error(f"调用失败: {e}")

# ===== 模式：政策合规检查 =====
elif mode == "⚖️ 政策合规检查":
    st.markdown("##### 让 AI 审查贷款审批是否符合监管要求")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("贷款金额", 1000, 500000, 30000)
        age = st.number_input("申请人年龄", 18, 80, 30)
    with col2:
        interest_rate = st.slider("利率(%)", 5.0, 30.0, 13.5)
        decision = st.selectbox("审批决策", ["通过", "拒绝", "人工复核"])
    with col3:
        purpose = st.selectbox("贷款用途", ["个人消费", "债务整合", "房地产投资"])
    
    if st.button("⚖️ 执行合规审查", type="primary"):
        try:
            from src.llm.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(client, prompt_mgr, configs.get("llm").agents.compliance_agent)
            features = {"loanAmnt": loan_amnt, "age": age, "purpose": purpose}
            
            with st.spinner("AI 审查中..."):
                result = agent.check(features=features, decision=decision, interest_rate=interest_rate)
            
            if "raw_response" in result:
                st.warning("LLM 返回了非 JSON 格式内容：")
                st.text(result["raw_response"])
            else:
                if result.get("is_compliant", False):
                    st.success("✅ 审查通过：该笔贷款符合监管要求。")
                else:
                    st.error("❌ 审查未通过：发现合规风险！")
                
                violations = result.get("violations", [])
                if violations:
                    st.markdown("### 🚨 违规项")
                    for v in violations:
                        st.error(f"**违反规则**: {v.get('rule')}\n\n**原因**: {v.get('reason')}")
                
                warnings = result.get("warnings", [])
                if warnings:
                    st.markdown("### ⚠️ 警告")
                    for w in warnings:
                        st.warning(w)
                        
                recommendations = result.get("recommendations", [])
                if recommendations:
                    st.markdown("### 💡 改进建议")
                    for r in recommendations:
                        st.info(r)
                        
        except Exception as e:
            st.error(f"调用失败: {e}")

# ==========================================
# 【改动点 5】把前面你粘错在文件末尾的两个 radio 删了，直接合并成了这个 elif
# ==========================================
elif mode == "🛡️ 预算与架构监控":
    st.markdown("##### 扩展功能：多厂商架构 与 Token 预算控制")
    
    st.markdown("### 1. 工厂模式架构（支持一键切换）")
    st.info("系统底层已实现工厂模式。如果 DeepSeek 不可用，在配置中将 `provider: deepseek` 改为 `provider: tongyi`，业务代码 0 修改即可直接切换到通义千问。")
    st.code("""
# llm_config.yaml
deepseek:
  provider: deepseek  # 支持 deepseek / tongyi
  model: "deepseek-chat"
""", language="yaml")
    
    st.markdown("### 2. 每日 Token 预算监控")
    budget_file = Path("./data/llm_budget.json")
    if budget_file.exists():
        try:
            data = json.loads(budget_file.read_text(encoding="utf-8"))
            used = data.get("used_tokens", 0)
            limit = 100000
            
            col1, col2 = st.columns(2)
            col1.metric("今日已消耗 Token", f"{used:,}")
            col2.metric("每日上限预算", f"{limit:,}")
            
            st.progress(min(used / limit, 1.0))
            st.caption(f"最后一次更新: {data.get('date', '')}")
            
            if used > limit * 0.8:
                st.warning("⚠️ 预算即将耗尽，请留意调用频率！")
        except Exception as e:
            st.error(f"读取预算文件失败: {e}")
    else:
        st.info("💡 今日尚未产生 Token 消耗。请去 `💬 自由对话` 聊两句，再切回这里看进度条。")