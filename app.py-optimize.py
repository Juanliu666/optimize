import os
import sys
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_api import predict

# 初始化数据存储
if 'result' not in st.session_state:
    st.session_state.result = None

# 设置页面配置
st.set_page_config(
    page_title="污泥-煤共热解产物预测系统",
    page_icon="🌱",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主背景色 */
    .stApp {
        background-color: #e6f2ff;
    }

    /* 子标题样式 */
    .custom-subheader {
        color: #1E90FF;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        border-bottom: 2.5px solid #1E90FF;
        padding-bottom: 1rem;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        width: 100%;
        padding: 0.75rem;
    }

    .stButton>button:hover {
        background-color: #0066CC;
        color: white;
    }

    /* 指标卡片样式 */
    .metric-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* 特别说明样式 */
    .instructions {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 列间距调整 */
    .stColumn {
        padding: 0 15px;
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🌱 污泥-煤共热解产物预测系统")
st.text("本系统基于注意力机制-长短时记忆网络算法（Attention-lstm）开发，左侧输入框中输入参数值，点击预测按钮进行预测计算，右侧区域将显示7项预测指标的结果。")

# 创建两列布局
col1, col2 = st.columns([1, 1])

# 左侧列 - 输入参数
with col1:
    #使用自定义子标题样式
    st.markdown('<h2 class="custom-subheader">输入参数</h2>', unsafe_allow_html=True)

    # 创建文本输入框
    input1 = st.text_input("污泥添加比例（%）", key="input1",
                           help="请输入0-100之间的数值")
    input2 = st.text_input("C含量（%）", key="input2",
                           help="请输入0-100之间的数值")
    input3 = st.text_input("热解温度（℃）", key="input3",
                           help="请输入500-900之间的数值")

    # 创建按钮
    if st.button("预测", key="predict_btn"):
        with st.spinner("预测中..."):
            # 验证输入
            try:
                # 转换为浮点数并验证范围
                val1 = float(input1)
                val2 = float(input2)
                val3 = float(input3)

                if not (0 <= val1 <= 100):
                    st.error("污泥添加比例必须在0-100之间")
                elif not (0 <= val2 <= 100):
                    st.error("C含量必须在0-100之间")
                elif not (500 <= val3 <= 900):
                    st.error("热解温度必须在500-900之间")
                else:
                    # 所有输入有效，进行预测
                    predict_param_list = [input1, input2, input3]
                    st.session_state.result = predict(predict_param_list)
            except ValueError:
                st.error("请输入有效的数字")
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧列 - 输出结果
with col2:
    # 使用自定义子标题样式
    st.markdown('<h2 class="custom-subheader">预测结果</h2>', unsafe_allow_html=True)

    # 结果展示区域
    if st.session_state.result:
        st.success("数据预测完成！")

        # 使用指标卡片展示结果
        predictions = st.session_state.result["predictions"]

        st.markdown(f'<div class="metric-card">气体中CH4含量（%）: {predictions["气体中CH4含量（%）"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">气体中CO2含量（%）: {predictions["气体中CO2含量（%）"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">气体产率（%）: {predictions["气体产率（%）"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">液体产率（%）: {predictions["液体产率（%）"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card">热解油中含氮化合物含量（%）: {predictions["热解油中含氮化合物含量（%）"]}</div>',
            unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">热解油中酚含量（%）: {predictions["热解油中酚含量（%）"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">热解油中酸含量（%）: {predictions["热解油中酸含量（%）"]}</div>',
                    unsafe_allow_html=True)

        # 原始数据查看
        with st.expander("查看原始JSON数据"):
            st.json(st.session_state.result)
    else:
        st.info("请输入参数并点击预测按钮获取结果")

    st.markdown('</div>', unsafe_allow_html=True)

# 特别说明
st.markdown('<div class="instructions">', unsafe_allow_html=True)
st.subheader("特别说明")
st.markdown("""
如需查看原始数据，可在预测结果处展开"查看原始JSON数据"部分
""")
st.markdown('</div>', unsafe_allow_html=True)
