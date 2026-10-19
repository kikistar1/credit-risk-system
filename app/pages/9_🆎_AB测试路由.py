import sys
from pathlib import Path
import streamlit as st
import hashlib

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="A/B 测试", page_icon="🆎", layout="wide")
render_sidebar()

st.title("🆎 A/B 测试路由模拟")
st.caption("演示基于一致性哈希的灰度发布与分流机制")

st.markdown("""
### 💡 原理说明
在生产环境中，我们需要测试新模型（B 组）是否比老模型（A 组）表现更好。
`ABTestRouter` 使用**一致性哈希算法**，确保**同一个用户每次访问都会被分配到同一个组**。
""")

st.divider()
st.markdown("### 🔬 分流测试")

col1, col2 = st.columns(2)
with col1:
    user_id = st.text_input("输入用户 ID (例如随便敲几个数字)", value="10086")
    ratio = st.slider("A 组分流比例", 0.0, 1.0, 0.5, 0.1)

if st.button("🚀 进行路由", type="primary"):
    if not user_id:
        st.error("请输入 User ID")
    else:
        # 一致性哈希计算
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100.0
        
        st.success(f"✅ 哈希值计算完成！分桶位置：{bucket:.2f}")
        
        if bucket < ratio:
            group = "A"
            st.info(f"当前走的是 **A 组模型 (基线)**")
        else:
            group = "B"
            st.info(f"当前走的是 **B 组模型 (新版)**")
            
        st.markdown(f"### 📊 路由结果")
        st.markdown(f"用户 **{user_id}** 稳定分配至：<h2 style='color:red;'>{group} 组</h2>", unsafe_allow_html=True)
        st.caption("💡 只要你不修改 User ID，反复点击按钮，结果永远一致。你可以尝试修改几位数字，看看分组是否会变化。")