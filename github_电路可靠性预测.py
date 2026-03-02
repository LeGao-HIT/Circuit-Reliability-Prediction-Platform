#####
#####



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SSA import SSA
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from scipy import stats, optimize
from scipy.stats import lognorm
from scipy.optimize import fsolve,minimize
import scipy.optimize as opt
from bayes_opt import BayesianOptimization
import time
from scipy.optimize import minimize, Bounds
from pathlib import Path
if 'first_part_completed' not in st.session_state:
    st.session_state['first_part_completed'] = False
# Streamlit 应用标题
st.title('📟电路可靠性预测')
st.subheader("一、数据选取")
# 文件上传
# 文件上传（可选）+ 默认加载仓库示例文件
file_path = st.sidebar.file_uploader("上传 Excel 文件（可选：上传后将覆盖示例数据）", type=['xlsx'])

# ✅ 默认示例文件路径：data/demo.xlsx（相对 app.py 所在目录）
DEMO_EXCEL = Path(__file__).resolve().parent / "测试数据.xlsx"

if file_path is None:
    if DEMO_EXCEL.exists():
        file_path = str(DEMO_EXCEL)  # 让后面代码保持不变（仍然用 file_path）
        st.sidebar.caption(f"当前数据源：{DEMO_EXCEL.name}（示例文件）")
    else:
        st.sidebar.error(f"未找到示例文件：{DEMO_EXCEL}。请上传 Excel 或将 demo.xlsx 放到 data/ 目录。")
        st.stop()
else:
    st.sidebar.caption(f"当前数据源：{file_path.name}（已上传）")


if file_path is not None:
    try:
        # 加载 Excel 文件的工作簿名称
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        # 让用户选择一个工作簿
        sheet_name = st.sidebar.selectbox("选择一个工作簿", sheet_names)

        # 添加模型选择器
        # 模型选择
        model_options = {
            "肖特基势垒二极管金半接触退化模型": "Schottky Barrier Diode Gold-Semiconductor Degradation Model",
            "肖特基势垒二极管金属化电迁移模型": "Schottky Barrier Diode Electromigration Model",
            "PN结整流二极管PN结特性退化模型": "PN Junction Rectifier Diode PN Junction Degradation Model",
            "PN结整流二极管热载流子注入模型": "PN Junction Rectifier Diode Hot Carrier Injection Model",
            "双极晶体管热载流子注入模型": "Bipolar Transistor Hot Carrier Injection Model",
            "双极晶体管PN结特性退化模型": "Existing Model"
        }
        selected_model = st.selectbox("选择一个模型", list(model_options.keys()))
        # 根据选择的模型调整图表标题和模型公式
        # 根据选择的模型调整图表标题和模型公式
        if selected_model == "肖特基势垒二极管金半接触退化模型":
            model_formula = r'''
                    $$
                    V_F = V_{F0} + A \times V_R^m \times e^{\frac{{-E_a}}{{K \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "肖特基势垒二极管金属化电迁移模型":
            model_formula = r'''
                    $$
                    V_F = V_{F0} + A \times I_R^n \times e^{\frac{{-E_a}}{{R \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管PN结特性退化模型":
            model_formula = r'''
                    $$
                    I_R = I_{R0} + A \times V_R^m \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆I_R'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管热载流子注入模型":
            model_formula = r'''
                    $$
                    I_R = I_{R0} + A \times I_F^n \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆I_R'
            chart_title = model_options[selected_model]

        elif selected_model == "双极晶体管热载流子注入模型":
            model_formula = r'''
                    $$
                    \beta = \beta_0 + A \times I_c^n \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆β'
            chart_title = model_options[selected_model]

        else:
            model_formula = r'''
                    $$
                    \beta = \beta_0 + A \times (\lvert V_R \rvert^m) \times e^{\frac{{-E_a}}{{K \times T}}} \times (t^p)
                    $$
                    '''
            ylabel = '∆β'
            chart_title = "数据分析和拟合"

        # 在 Streamlit 应用中显示选定的模型公式
        st.write("模型公式：")
        st.markdown(model_formula, unsafe_allow_html=True)

        # 读取选定的工作簿数据
        df = xls.parse(sheet_name)

        with st.expander("**数据在线编辑：**"):
            # 显示 DataFrame
            # 使用 AgGrid 创建可编辑的表格
            grid = AgGrid(
                df,
                editable=True,  # 启用编辑功能
                height=400,  # 设置表格高度
                width='100%',  # 设置表格宽度为100%

            )

            # 获取编辑后的 DataFrame
            updated_df = grid['data']
        st.write("编辑后数据：")
        st.dataframe(updated_df)
        for column in updated_df.columns:
            # 尝试将每一列转换为数值类型
            updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce')
        #st.write(updated_df.dtypes)

        #st.write('updated_df', updated_df)
        df = updated_df.copy()

        #st.write('df', df)
        #st.dataframe(df)

    except Exception as e:
        st.error(f"处理文件时发生错误：{e}")
    # 使用 Streamlit 的交互性组件来获取用户输入
    st.sidebar.header("设置输入参数")
    # 添加复选框和条件性数值输入

    # 用户输入行和列范围的示例格式："起始行:结束行, 起始列:结束列"
    # 使用 Streamlit 的交互性组件来分组设置应力参数
    with st.sidebar.expander("应力条件1的数据范围"):
        range_input_1 = st.text_input("条件1数据范围", "2:11, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_1 = st.number_input("应力1.1", value=398.000, format="%.4f",help="开尔文温度")
        VR1 = st.number_input("应力1.2", value=200.000, format="%.4f")

    with st.sidebar.expander("应力条件2的数据范围"):
        range_input_2 = st.text_input("条件2数据范围", "12:21, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_2 = st.number_input("应力2.1", value=398.000, format="%.4f",help="都是卡尔文温度")
        VR2 = st.number_input("应力2.2", value=600.000, format="%.4f")

    with st.sidebar.expander("应力条件3的数据范围"):
        range_input_3 = st.text_input("条件3数据范围", "22:31, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_3 = st.number_input("应力3.1", value=398.000, format="%.4f")
        VR3 = st.number_input("应力3.2", value=600.000, format="%.4f")
    with st.sidebar.expander("应力条件4的数据范围"):
        range_input_4 = st.text_input("条件4数据范围", "36:45, 3:18", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_4 = st.number_input("应力4.1", value=423.000, format="%.4f")
        VR4 = st.number_input("应力4.2", value=600.000, format="%.4f")
    def parse_range(range_str):
        # 解析行和列范围
        row_range, col_range = range_str.split(',')
        start_row, end_row = map(int, row_range.split(':'))
        start_col, end_col = map(int, col_range.split(':'))
        return start_row, end_row, start_col, end_col
    # 解析用户输入的范围
    start_row_1, end_row_1, start_col_1, end_col_1 = parse_range(range_input_1)
    start_row_2, end_row_2, start_col_2, end_col_2 = parse_range(range_input_2)
    start_row_3, end_row_3, start_col_3, end_col_3 = parse_range(range_input_3)
    start_row_4, end_row_4, start_col_4, end_col_4 = parse_range(range_input_4)

    # 为四组 t 数据设置行和列范围的输入
    with st.expander("应力条件1时间"):
        t1_row_index = st.number_input("t1 数据所在的行索引", value=1, min_value=0)
        t1_start_col_index = st.number_input("t1 数据的起始列索引", value=3, min_value=0)
        t1_end_col_index = st.number_input("t1 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件2时间"):
        t2_row_index = st.number_input("t2 数据所在的行索引", value=1, min_value=0)
        t2_start_col_index = st.number_input("t2 数据的起始列索引", value=3, min_value=0)
        t2_end_col_index = st.number_input("t2 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件3时间"):
        t3_row_index = st.number_input("t3 数据所在的行索引", value=1, min_value=0)
        t3_start_col_index = st.number_input("t3 数据的起始列索引", value=3, min_value=0)
        t3_end_col_index = st.number_input("t3 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件4时间"):
        t4_row_index = st.number_input("t4 数据所在的行索引", value=35, min_value=0)
        t4_start_col_index = st.number_input("t4 数据的起始列索引", value=3, min_value=0)
        t4_end_col_index = st.number_input("t4 数据的结束列索引", value=18, min_value=0)

    st.subheader("二、数据处理")
    def preprocess_data(beta_data, alpha):
        # 计算 IQR
        Q1 = beta_data.quantile(0.25)
        Q3 = beta_data.quantile(0.75)
        IQR = Q3 - Q1
        # 定义异常值的范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 过滤异常值
        filtered_df = beta_data.where(~((beta_data < lower_bound) | (beta_data > upper_bound)))
        # 将 NaN 值替换为每列的平均值
        filtered_df = filtered_df.fillna(filtered_df.mean())
        # 数据平滑处理
        smoothed_df = filtered_df.ewm(alpha=alpha).mean()
        return filtered_df, smoothed_df

    # 使用 Streamlit 的侧边栏创建一个滑块来调节 alpha 值
    alpha = st.number_input("输入平滑系数 alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.1,help="自动使用四分位数法（IQR）进行异常值剔除，此处使用指数加权平均法进行数据降噪处理")

    beta_1 = df.iloc[start_row_1:end_row_1 + 1, start_col_1:end_col_1 + 1]
    beta_2 = df.iloc[start_row_2:end_row_2 + 1, start_col_2:end_col_2 + 1]
    beta_3 = df.iloc[start_row_3:end_row_3 + 1, start_col_3:end_col_3 + 1]
    beta_4 = df.iloc[start_row_4:end_row_4 + 1, start_col_4:end_col_4 + 1]
    beta_1_0 = beta_1
    beta_2_0 = beta_2
    beta_3_0 = beta_3
    beta_4_0 = beta_4
    first_column1 = beta_1.iloc[:, 0]
    first_column2 = beta_2.iloc[:, 0]
    first_column3 = beta_3.iloc[:, 0]
    first_column4 = beta_4.iloc[:, 0]

    # 计算 Δbeta_1：每一列减去第一列
    #beta_1_1是求的差值
    beta_1_1 = beta_1.apply(lambda col: col - first_column1)
    beta_2_2 = beta_2.apply(lambda col: col - first_column2)
    beta_3_3 = beta_3.apply(lambda col: col - first_column3)
    beta_4_4 = beta_4.apply(lambda col: col - first_column4)


    #st.write('beta_1', beta_1)


    #st.write('beta_1',beta_1)

    beta_1_filtered, beta_1_smoothed = preprocess_data(beta_1, alpha)
    beta_2_filtered, beta_2_smoothed = preprocess_data(beta_2, alpha)
    beta_3_filtered, beta_3_smoothed = preprocess_data(beta_3, alpha)
    beta_4_filtered, beta_4_smoothed = preprocess_data(beta_4, alpha)

    # 获取第一列作为基准列
    smoothed1_first_column = beta_1_smoothed.iloc[:, 0]
    smoothed2_first_column = beta_2_smoothed.iloc[:, 0]
    smoothed3_first_column = beta_3_smoothed.iloc[:, 0]
    smoothed4_first_column = beta_4_smoothed.iloc[:, 0]
    # 计算 Δbeta_1：每一列减去第一列
    smoothed1 = beta_1_smoothed.apply(lambda col: col - smoothed1_first_column)
    smoothed2 = beta_2_smoothed.apply(lambda col: col - smoothed2_first_column)
    smoothed3 = beta_3_smoothed.apply(lambda col: col - smoothed3_first_column)
    smoothed4 = beta_4_smoothed.apply(lambda col: col - smoothed4_first_column)
    # 合并数据
    smoothed_data = {
        'stress_1_smoothed': beta_1_smoothed,
        'stress_2_smoothed': beta_2_smoothed,
        'stress_3_smoothed': beta_3_smoothed,
        'stress_4_smoothed': beta_4_smoothed
    }
    combined_df = pd.concat(smoothed_data,axis=1)

    # 转换为CSV
    csv = combined_df.to_csv(index=False).encode('utf-8')

    # 添加下载按钮
    st.download_button(
        label="下载数据",
        data=csv,
        file_name="smoothed_data.csv",
        mime="text/csv",
        help="点击此按钮下载处理后的数据"
    )


    # 提取四组 t 数据
    t1 = df.iloc[t1_row_index, t1_start_col_index:t1_end_col_index + 1].to_numpy()
    t2 = df.iloc[t2_row_index, t2_start_col_index:t2_end_col_index + 1].to_numpy()
    t3 = df.iloc[t3_row_index, t3_start_col_index:t3_end_col_index + 1].to_numpy()
    t4 = df.iloc[t4_row_index, t4_start_col_index:t4_end_col_index + 1].to_numpy()

    #st.write(t)
    #st.write(t)
    #st.write('beta_1_smoothed',beta_1_smoothed)
    # 选择框，让用户选择要显示的数据类型
    st.write('**预处理后结果：**')
    option = st.selectbox(
        '选择要显示的数据集',
        ('原始数据和平滑后的数据对比', '原始数据', '平滑后的数据'))
    # 用户选择是否显示图例
    # show_legend = st.checkbox("显示图例", value=True)
    # 绘制图形
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 根据用户选择的数据类型进行绘制
    if option == '原始数据':
        axs[0, 0].plot(t1, beta_1_1.T, 'o',)
        axs[0, 1].plot(t2, beta_2_2.T, 'o', )
        axs[1, 0].plot(t3, beta_3_3.T, 'o', )
        axs[1, 1].plot(t4, beta_4_4.T, 'o', )
    elif option == '平滑后的数据':
        axs[0, 0].plot(t1, smoothed1.T, 'o-', alpha=0.8)
        axs[0, 1].plot(t2, smoothed2.T, 'o-', alpha=0.8)
        axs[1, 0].plot(t3, smoothed3.T, 'o-', alpha=0.8 )
        axs[1, 1].plot(t4, smoothed4.T, 'o-', alpha=0.8)
    elif option == '原始数据和平滑后的数据对比':
        axs[0, 0].plot(t1, beta_1_1.T, 'o',)
        axs[0, 0].plot(t1, smoothed1.T, '-',)
        axs[0, 1].plot(t2, beta_2_2.T, 'o', label='2',alpha=0.6)
        axs[0, 1].plot(t2, smoothed2.T, '-', )
        axs[1, 0].plot(t3, beta_3_3.T, 'o', label='22',alpha=0.6)
        axs[1, 0].plot(t3, smoothed3.T, '-', )
        axs[1, 1].plot(t4, beta_4_4.T, 'o' , label='2',alpha=0.6)
        axs[1, 1].plot(t4, smoothed4.T, '-', )
    # 设置图例和图形标题
    for ax in axs.flat:
        #if show_legend:
           # ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        #ax.set_title(f'{chart_title} ')
        #ax.grid(True)
    # 设置子图标题
    axs[0, 0].set_title('Stress condition 1 ')
    axs[0, 1].set_title('Stress condition 2 ')
    axs[1, 0].set_title('Stress condition 3 ')
    axs[1, 1].set_title('Stress condition 4 ')
    # 调整布局
    plt.tight_layout()
    # 显示图形
    st.pyplot(fig)

    # 具体应力的输入1-4
    beta_1 = beta_1_smoothed.to_numpy()
    beta_2 = beta_2_smoothed.to_numpy()
    beta_3 = beta_3_smoothed.to_numpy()
    beta_4 = beta_4_smoothed.to_numpy()

    #st.write('beta_1',beta_1)
    #st.write(beta_175)


    # Beta_0 列索引
    # 使用 Streamlit 的交互性组件来分组设置 Beta_0 的列索引
    with st.sidebar.expander("设置t=0列索引（初始时刻的数据）"):
        # 让用户输入 beta_0_150 所在的列索引
        beta_0_1_col_index = st.number_input("应力1 t=0所在的列索引", value=3, min_value=0)

        # 让用户输入 beta_0_175 所在的列索引
        beta_0_2_col_index = st.number_input("应力2 t=0所在的列索引", value=3, min_value=0)
        beta_0_3_col_index = st.number_input("应力3 t=0所在的列索引", value=3, min_value=0)
        beta_0_4_col_index = st.number_input("应力4 t=0所在的列索引", value=3, min_value=0)
    # 根据用户输入的列索引获取 beta_0 值
    beta_0_1 = df.iloc[start_row_1:end_row_1 + 1, beta_0_1_col_index].mean()
    beta_0_2 = df.iloc[start_row_2:end_row_2 + 1, beta_0_2_col_index].mean()
    beta_0_3 = df.iloc[start_row_3:end_row_3 + 1, beta_0_3_col_index].mean()
    beta_0_4 = df.iloc[start_row_4:end_row_4 + 1, beta_0_4_col_index].mean()

    #beta_0=(beta_0_1+beta_0_2+beta_0_3+beta_0_4)/4


    #st.write('beta_0' , beta_0)
    #st.write(beta_0_175)
    # 将摄氏温度转换为开尔文
    T_1 = T_C_1
    T_2 = T_C_2
    T_3 = T_C_3
    T_4 = T_C_4
    # 根据用户输入的索引获取 t 数据

    # 玻尔兹曼常数，单位 eV/K
    K = 8.617333262145e-5

    formula_type = st.radio("选择所用数据的类型", ("正常数据", "增量型数据"),help=("增量型数据：指的是每个数据都跟所选数据的第一列（0时刻的初始值）做差"))

    if formula_type == "增量型数据":
        beta_1=smoothed1
        beta_2=smoothed2
        beta_3=smoothed3
        beta_4=smoothed4

        beta_0_1=0
        beta_0_2=0
        beta_0_3=0
        beta_0_4=0
    else:
        if selected_model == "肖特基势垒二极管金半接触退化模型":
            ylabel = 'V_F'

        elif selected_model == "肖特基势垒二极管金属化电迁移模型":
            ylabel = 'V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管PN结特性退化模型":
            ylabel = 'I_R'

        elif selected_model == "PN结整流二极管热载流子注入模型":
            ylabel = 'I_R'

        elif selected_model == "双极晶体管热载流子注入模型":
            ylabel = 'β'

        else:
            ylabel = 'β'


    st.subheader("三、元器件可靠性建模")
    st.markdown("#### 3.1、定义搜索空间")
    fix_ea = st.checkbox('固定Ea')
    if fix_ea:
        ea_value = st.number_input("输入 E_a 的值", value=0.50, min_value=0.00, max_value=10.00, step=0.01,
                                           format="%.2f")
    # 使用 Streamlit 的交互性组件来分组设置参数
    with st.expander("设置 A 的搜索范围"):
        a_min = st.number_input("A 的最小值", value=-1000.00, format="%.2f")
        a_max = st.number_input("A 的最大值", value=1000.00, format="%.2f")

    with st.expander("设置 m 的搜索范围"):
        m_min = st.number_input("m 的最小值", value=2.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
        m_max = st.number_input("m 的最大值", value=4.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
    if not fix_ea:
        with st.expander("设置 E_a 的搜索范围"):
            ea_min = st.number_input("E_a 的最小值", value=0.10, min_value=0.00, max_value=10.00, step=0.10,
                                     format="%.2f")
            ea_max = st.number_input("E_a 的最大值", value=1.00, min_value=0.00, max_value=10.00, step=0.10,
                                     format="%.2f")

    with st.expander("设置 p 的搜索范围"):
        p_min = st.number_input("p 的最小值", value=0.50, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
        p_max = st.number_input("p 的最大值", value=2.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")

    if fix_ea:
        # 固定 E_a 时的搜索空间，只包括 A、m 和 p
        E_a = ea_value
        search_space = {
            0: (a_min, a_max),  # A 的搜索范围
            1: (m_min, m_max),  # m 的搜索范围
            2: (p_min, p_max)  # p 的搜索范围
        }
        n_dim = 3  # 参数数量减少为 3
        def objective_function(params, beta_0, T, VR, beta, t):
            A, m,  p = params
            predicted_beta = beta_0 + A * (VR ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            mse = np.mean((beta - predicted_beta) ** 2)
            return mse

        def function(params, beta_0, T, VR, t):
            A, m, p = params
            predicted_beta = beta_0 + A * (np.abs(VR) ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            return predicted_beta

        def fitness_function(params):
            loss_1 = objective_function(params, beta_0_1, T_1, VR1, beta_1, t1)
            loss_2 = objective_function(params, beta_0_2, T_2, VR2, beta_2, t2)
            loss_3 = objective_function(params, beta_0_3, T_3, VR3, beta_3, t3)
            loss_4 = objective_function(params, beta_0_4, T_4, VR4, beta_4, t4)
            total_loss = loss_1 + loss_2 + loss_3 + loss_4
            return total_loss  # 在最小化问题中，适应度越低越好
    else:

        n_dim = 4
        # 根据用户输入定义搜索空间
        search_space = {
            0: (a_min, a_max),  # A 的搜索范围
            1: (m_min, m_max),  # m 的搜索范围
            2: (ea_min, ea_max),  # E_a 的搜索范围
            3: (p_min, p_max)  # p 的搜索范围
        }
        def objective_function(params, beta_0, T, VR, beta, t):
            A, m, E_a, p = params
            predicted_beta = beta_0 + A * (VR ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            mse = np.mean((beta - predicted_beta) ** 2)
            return mse

        def function(params, beta_0, T, VR, t):
            A, m, E_a, p = params
            predicted_beta = beta_0 + A * (np.abs(VR) ** m) * np.exp(-(E_a / (K * T))) * (t ** p)

            return predicted_beta

        def fitness_function(params):
            loss_1 = objective_function(params, beta_0_1, T_1, VR1, beta_1, t1)
            loss_2 = objective_function(params, beta_0_2, T_2, VR2, beta_2, t2)
            loss_3 = objective_function(params, beta_0_3, T_3, VR3, beta_3, t3)
            loss_4 = objective_function(params, beta_0_4, T_4, VR4, beta_4, t4)
            total_loss = loss_1 + loss_2 + loss_3 + loss_4
            return total_loss  # 在最小化问题中，适应度越低越好

    with st.sidebar.expander("均值和标准差（生成正态分布）"):
            # 使用st.sidebar.number_input来接收用户输入的均值和标准差
            mu = st.number_input("请输入均值:", value=0.91)
            sd = st.number_input("请输入标准差:", value=0.02)
            st.caption("说明：μ/σ 用于生成器件初值样本（Monte Carlo），用于计算寿命分布与 R(t)、F(t)。")

    st.sidebar.header("算法参数设置")
    # 让用户调整搜索算法中的个体数目，影响搜索的广度和速度
    pop_size = st.sidebar.number_input("搜索个体数目 (种群大小)",
                                 min_value=10,
                                 max_value=2000,
                                 value=100,
                                 help="调整个体数目来影响算法的搜索范围和速度。较大的数目可能提高找到最优解的概率，但会增加计算量。")

    # 让用户调整算法运行的迭代次数，影响搜索的深度和精确度
    max_iter = st.sidebar.number_input("搜索迭代次数 (最大迭代次数)",
                                 min_value=10,
                                 max_value=2000,
                                 value=100,
                                 help="调整迭代次数来影响算法的搜索深度和精确度。较多的迭代次数可能提高解的质量，但会增加运行时间。")




    #accuracy_percentage = st.sidebar.number_input("输入显示精度 (例如输入 80 代表 ±20%)", value=80,                                                  min_value=50, max_value=100)



    #mu = 130.61737499999998
    #sd = 18.85211

    if st.button('**运行参数拟合**',help="点此按钮进行模型参数拟合"):
        try:
            ssa = SSA(fitness_function, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, search_space=search_space)
            best_params = ssa.run()
            best_params_values = ssa.gbest_x
            st.session_state['first_part_completed'] = True
            # 将参数和结果显示在 Streamlit 应用上
            best_params_values = [f"{param:.9e}" for param in best_params_values]

            if fix_ea:
                # 固定 E_a 时的参数结果展示
                best_params_dict = {
                    "A": best_params_values[0],
                    "m": best_params_values[1],
                    "p": best_params_values[2],
                    "E_a (固定)": ea_value
                }
            else:
                # 不固定 E_a 时的参数结果展示
                best_params_dict = {
                    "A": best_params_values[0],
                    "m": best_params_values[1],
                    "E_a": best_params_values[2],
                    "p": best_params_values[3]
                }


            best_params_df = pd.DataFrame(best_params_dict, index=[0])
            pd.options.display.float_format = '{:.4f}'.format

            st.markdown("#### 3.2、运行结果")

            st.write("最佳参数：",best_params_df)
            #st.dataframe(best_params_df)
            # 显示最佳损失
            st.write("最佳损失:", ssa.gbest_y)
            # 绘制优化过程图
            st.write("训练损失：")
            fig, ax = plt.subplots()
            ax.plot(ssa.gbest_y_hist)
            ax.set_title('Optimization process')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            st.pyplot(fig)
            # 使用最佳参数计算预测值
            params = ssa.gbest_x
            predicted_beta_1 = function(params, beta_0_1, T_1, VR1, t1)
            predicted_beta_2 = function(params, beta_0_2, T_2, VR2, t2)
            predicted_beta_3 = function(params, beta_0_3, T_3, VR1, t3)
            predicted_beta_4 = function(params, beta_0_4, T_4, VR2, t4)

            # 计算95%置信区间
            z_score = 1.96  # 95%置信区间的Z分数
            #conf_interval_upper = mu + (z_score * sd)
            #conf_interval_lower = mu - (z_score * sd)







            # 用 delta_predicted_beta_1 来计算置信区间
            delta_predicted_beta_1_upper = predicted_beta_1 + (z_score * sd)
            delta_predicted_beta_1_lower = predicted_beta_1 - (z_score * sd)
            delta_predicted_beta_2_upper = predicted_beta_2 + (z_score * sd)
            delta_predicted_beta_2_lower = predicted_beta_2 - (z_score * sd)
            delta_predicted_beta_3_upper = predicted_beta_3 + (z_score * sd)
            delta_predicted_beta_3_lower = predicted_beta_3 - (z_score * sd)
            delta_predicted_beta_4_upper = predicted_beta_4 + (z_score * sd)
            delta_predicted_beta_4_lower = predicted_beta_4 - (z_score * sd)

            # 计算 predicted_beta_1 在 x=0 时的值
            #beta_1_at_0 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_1 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_2 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_3 = function(params, beta_0, T_1, VR1, 0)


            st.write("**拟合结果图像：**")
            # 绘制比较图
            fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
            #st.write(beta_1)
            #st.write(beta_1.T)
            #st.write(predicted_beta_150)


            axs1[0,0].plot(t1, beta_1.T, 'o')
            axs1[0,0].plot(t1, predicted_beta_1, 'r-', label='Fitting ')
            axs1[0,0].plot(t1, delta_predicted_beta_1_upper, 'r--', label='95% confidence interval')
            axs1[0,0].plot(t1, delta_predicted_beta_1_lower, 'r--')
            axs1[0,0].set_xlabel('Time')
            axs1[0,0].set_ylabel(ylabel)
            axs1[0,0].set_title('Stress 1')
            axs1[0,0].legend()
            #axs1[0, 0].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0),                               )
            #st.write(predicted_beta_1)
            axs1[0,1].plot(t2, beta_2.T, 'o')
            axs1[0,1].plot(t2, predicted_beta_2, 'r-', label='Fitting')
            axs1[0,1].plot(t2, delta_predicted_beta_2_upper, 'r--', label='95% confidence interval')
            axs1[0,1].plot(t2, delta_predicted_beta_2_lower, 'r--')
            axs1[0,1].set_xlabel('Time')
            axs1[0,1].set_ylabel(ylabel)
            axs1[0,1].set_title('Stress 2')
            axs1[0,1].legend()
            #axs1[0, 1].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[1,0].plot(t3, beta_3.T, 'o')
            axs1[1,0].plot(t3, predicted_beta_3, 'r-', label='Fitting ')
            axs1[1,0].plot(t3, delta_predicted_beta_3_upper, 'r--', label='95% confidence interval')
            axs1[1,0].plot(t3, delta_predicted_beta_3_lower, 'r--')
            axs1[1,0].set_xlabel('Time')
            axs1[1,0].set_ylabel(ylabel)
            axs1[1,0].set_title('Stress 3')
            axs1[1,0].legend()
            #axs1[1, 0].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[1,1].plot(t4, beta_4.T, 'o')
            axs1[1,1].plot(t4, predicted_beta_4, 'r-', label='Fitting ')
            axs1[1,1].plot(t4, delta_predicted_beta_4_upper, 'r--', label='95% confidence interval')
            axs1[1,1].plot(t4, delta_predicted_beta_4_lower, 'r--')
            axs1[1,1].set_xlabel('Time')
            axs1[1,1].set_ylabel(ylabel)
            axs1[1,1].set_title('Stress 4')
            axs1[1,1].legend()
            #axs1[1, 1].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[0, 0].set_xlim(left=0)
            axs1[0, 1].set_xlim(left=0)
            axs1[1, 0].set_xlim(left=0)
            axs1[1, 1].set_xlim(left=0)
            # 显示预测值


            plt.tight_layout()
            st.pyplot(fig1)
            st.session_state['params'] = ssa.gbest_x
            #st.session_state['first_part_completed'] = True
            # 创建一个 DataFrame
            predicted_beta_df1 = pd.DataFrame({
                'Stress 1': predicted_beta_1,
            },index=t1)
            predicted_beta_df2 = pd.DataFrame({
                'Stress 2': predicted_beta_2,
            }, index=t2)
            predicted_beta_df3 = pd.DataFrame({
                'Stress 3': predicted_beta_3,
            }, index=t3)
            predicted_beta_df4 = pd.DataFrame({
                'Stress 4': predicted_beta_4,
            }, index=t4)
            # 显示 DataFrame
            st.write("Stress 1 预测结果:", predicted_beta_df1.T)
            st.write("Stress 2 预测结果:", predicted_beta_df2.T)
            st.write("Stress 3 预测结果:", predicted_beta_df3.T)
            st.write("Stress 4 预测结果:", predicted_beta_df4.T)
        except Exception as e:
            st.error(f"运行模型时发生错误：{e}")
##############################
    st.subheader("四、元器件可靠度求解")
    st.markdown("#### 4.1、参数设置")

    if 'params' in st.session_state and st.session_state['params'] is not None:
        params0 = pd.DataFrame({
            '拟合参数': st.session_state['params'],
        })
        st.write('使用模型参数（A、m、(E_a)、p）:', params0.T)
    else:
        st.write('**通知**：尚未进行参数拟合。')

    st.write('注意：通过**运行参数拟合**按钮，可以对模型参数进行更新。')

    o = st.number_input("定义失效阈值：", value=0.95)
    ff = st.number_input("定义图像横坐标显示范围：", value=2000)


    # ✅ 只保留一个按钮：可靠度求解
    reliability_clicked = st.button('**可靠度求解**', help="点击此按钮直接求解可靠度")

    if reliability_clicked:
        try:
            # 生成随机初值 beta0（正态分布）
            beta0 = np.random.normal(mu, sd, 500)

            params = st.session_state['params']

            # 四组寿命数组
            lifetime1 = np.zeros(len(beta0))
            lifetime2 = np.zeros(len(beta0))
            lifetime3 = np.zeros(len(beta0))
            lifetime4 = np.zeros(len(beta0))


            # 失效方程：function(params, ...) - 阈值 = 0
            def equation(t, beta0_i, T, VR):
                return function(params, beta0_i, T, VR, t) - o


            # 求解每个样本的寿命（用多个初值避免局部根）
            initial_guesses = [1, 10, 100, 1000]
            for i in range(len(beta0)):
                # condition 1
                sols = [fsolve(equation, guess, args=(beta0[i], T_1, VR1))[0] for guess in initial_guesses]
                lifetime1[i] = max(sols) if np.std(sols) > 1e-5 else sols[0]

                # condition 2
                sols = [fsolve(equation, guess, args=(beta0[i], T_2, VR2))[0] for guess in initial_guesses]
                lifetime2[i] = max(sols) if np.std(sols) > 1e-5 else sols[0]

                # condition 3
                sols = [fsolve(equation, guess, args=(beta0[i], T_3, VR3))[0] for guess in initial_guesses]
                lifetime3[i] = max(sols) if np.std(sols) > 1e-5 else sols[0]

                # condition 4
                sols = [fsolve(equation, guess, args=(beta0[i], T_4, VR4))[0] for guess in initial_guesses]
                lifetime4[i] = max(sols) if np.std(sols) > 1e-5 else sols[0]

            # 画可靠度曲线 + 累计失效概率曲线
            fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))  # R(t)
            fig3, axs3 = plt.subplots(2, 2, figsize=(10, 10))  # F(t)

            import matplotlib.ticker as ticker

            lifetimes = [lifetime1, lifetime2, lifetime3, lifetime4]

            for idx, lifetime in enumerate(lifetimes):
                row = idx // 2
                col = idx % 2

                # 拟合对数正态分布
                shape, loc, scale = stats.lognorm.fit(lifetime, floc=0)

                x = np.linspace(0, ff, 10001)

                # ✅ 累计失效概率：F(t) = CDF
                F_t = stats.lognorm.cdf(x, shape, loc, scale)

                # ✅ 可靠度：R(t) = 1 - F(t)
                R_t = 1 - F_t

                # --------- 画 R(t) ----------
                axs2[row, col].plot(x, R_t)
                axs2[row, col].set_title(f"Condition {idx + 1} — R(t)")
                axs2[row, col].set_xlabel("Time(h)")
                axs2[row, col].set_ylabel("R(t)")
                axs2[row, col].yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda val, pos: f'{val:,.10f}')
                )

                # --------- 画 F(t) ----------
                axs3[row, col].plot(x, F_t)
                axs3[row, col].set_title(f"Condition {idx + 1} — F(t)")
                axs3[row, col].set_xlabel("Time(h)")
                axs3[row, col].set_ylabel("F(t)")
                axs3[row, col].yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda val, pos: f'{val:,.10f}')
                )

                # 标注：t_95 (R=0.95) 等价于 F=0.05
                t_95_indices = np.where(R_t <= 0.95)[0]
                if len(t_95_indices) > 0:
                    t_95 = x[t_95_indices[0]]

                    # R(t) 标注
                    axs2[row, col].axvline(x=t_95, color='red', linestyle='--', label=f't(0.95) = {t_95:.2f}')
                    axs2[row, col].axhline(y=0.95, color='red', linestyle='--')
                    axs2[row, col].legend()

                    # F(t) 标注（对应 F=0.05）
                    axs3[row, col].axvline(x=t_95, color='red', linestyle='--', label=f't(F=0.05) = {t_95:.2f}')
                    axs3[row, col].axhline(y=0.05, color='red', linestyle='--')
                    axs3[row, col].legend()

            plt.tight_layout()
            st.markdown("#### 4.2、可靠度函数图像 R(t)：")
            st.pyplot(fig2)

            plt.tight_layout()
            st.markdown("#### 4.3、累计失效概率图像 F(t) = 1 - R(t)：")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"处理分布时发生错误：{e}")

    # =======================
    # 五、电路级可靠性求解
    # =======================
    st.subheader("五、电路级可靠性求解")

    import matplotlib.patches as patches


    def clamp01(x):
        try:
            x = float(x)
        except:
            return None
        return max(0.0, min(1.0, x))


    def compute_series(R_list):
        R_list = [clamp01(r) for r in R_list]
        if any(r is None for r in R_list):
            return None
        R = 1.0
        for r in R_list:
            R *= r
        return R


    def compute_parallel(R_list):
        R_list = [clamp01(r) for r in R_list]
        if any(r is None for r in R_list):
            return None
        prod = 1.0
        for r in R_list:
            prod *= (1.0 - r)
        return 1.0 - prod


    def draw_series(n, title="Series Structure"):
        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.set_title(title)
        ax.axis("off")

        x0, y0 = 0.05, 0.45
        w, h = 0.08, 0.25
        gap = 0.04

        # line
        ax.plot([x0 - 0.03, x0 + n * (w + gap)], [y0 + h / 2, y0 + h / 2])

        for i in range(n):
            x = x0 + i * (w + gap)
            rect = patches.Rectangle((x, y0), w, h, fill=False, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y0 + h / 2, f"C{i + 1}", ha="center", va="center", fontsize=10)

        st.pyplot(fig)


    def draw_parallel(n, title="Parallel Structure"):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.set_title(title)
        ax.axis("off")

        xL, xR = 0.12, 0.88
        y_top, y_bot = 0.80, 0.20

        # left and right bus
        ax.plot([xL, xL], [y_bot, y_top], linewidth=2)
        ax.plot([xR, xR], [y_bot, y_top], linewidth=2)

        # branches
        ys = np.linspace(y_top, y_bot, n)
        w, h = 0.10, 0.12
        for i, y in enumerate(ys):
            # wires to component
            ax.plot([xL, 0.40], [y, y], linewidth=2)
            ax.plot([0.60, xR], [y, y], linewidth=2)

            rect = patches.Rectangle((0.40, y - h / 2), w, h, fill=False, linewidth=2)
            ax.add_patch(rect)
            ax.text(0.40 + w / 2, y, f"C{i + 1}", ha="center", va="center", fontsize=10)

        st.pyplot(fig)


    def draw_mixed(stages, title="Mixed Structure (Series of Stages)"):
        """
        stages: list of dict
          {"type": "single", "n": 1} or {"type": "parallel", "n": k}
        """
        fig, ax = plt.subplots(figsize=(10, 3.0))
        ax.set_title(title)
        ax.axis("off")

        x = 0.05
        y_mid = 0.55
        stage_w = 0.16
        gap = 0.05

        # main line
        ax.plot([0.02, 0.98], [y_mid, y_mid], linewidth=2)

        for si, stg in enumerate(stages):
            # stage bounding box (light)
            ax.add_patch(patches.Rectangle((x, 0.25), stage_w, 0.60, fill=False, linewidth=1, linestyle="--"))
            ax.text(x + stage_w / 2, 0.87, f"Stage {si + 1}", ha="center", va="center", fontsize=9)

            if stg["type"] == "single":
                # draw one component in the middle
                w, h = 0.07, 0.18
                rect = patches.Rectangle((x + stage_w / 2 - w / 2, y_mid - h / 2), w, h, fill=False, linewidth=2)
                ax.add_patch(rect)
                ax.text(x + stage_w / 2, y_mid, f"C", ha="center", va="center", fontsize=10)

            else:
                # parallel group inside this stage
                n = stg["n"]
                ys = np.linspace(0.75, 0.35, n)
                # local buses
                xL = x + 0.03
                xR = x + stage_w - 0.03
                ax.plot([xL, xL], [0.35, 0.75], linewidth=2)
                ax.plot([xR, xR], [0.35, 0.75], linewidth=2)

                w, h = 0.06, 0.12
                for bi, yy in enumerate(ys):
                    ax.plot([xL, x + stage_w / 2 - 0.03], [yy, yy], linewidth=2)
                    ax.plot([x + stage_w / 2 + 0.03, xR], [yy, yy], linewidth=2)
                    rect = patches.Rectangle((x + stage_w / 2 - w / 2, yy - h / 2), w, h, fill=False, linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + stage_w / 2, yy, f"C{bi + 1}", ha="center", va="center", fontsize=9)

            x += stage_w + gap

        st.pyplot(fig)


    # -----------------------
    # 5.1 串并联模型选择 + 结构图
    # -----------------------
    st.markdown("#### 5.1 串并联模型选择")

    model_type = st.selectbox(
        "选择电路拓扑结构",
        ["串联模型", "并联模型", "混合模型（串联若干级，每级可并联）"]
    )

    # 任务时间（可选：只是展示用途，可靠度输入默认就是该时刻的R）
    t_mission = st.number_input("任务时间/评估时刻（可选，仅用于标注）", value=0.0, min_value=0.0, format="%.4f")

    # -----------------------
    # 输入元器件可靠度 + 计算
    # -----------------------
    st.markdown("#### 5.2 电路可靠性求解结果")

    if model_type == "串联模型":
        n = st.number_input("串联元器件数量 n", min_value=1, max_value=200, value=4, step=1)
        draw_series(int(n), title="Series Structure")

        st.write("请输入每个元器件在任务时刻的可靠度 R_i（0~1）：")
        cols = st.columns(4)
        R_list = []
        for i in range(int(n)):
            with cols[i % 4]:
                Ri = st.number_input(f"R{i + 1}", min_value=0.0, max_value=1.0, value=0.99, format="%.6f")
            R_list.append(Ri)

        R_sys = compute_series(R_list)

    elif model_type == "并联模型":
        n = st.number_input("并联支路元器件数量 n", min_value=1, max_value=50, value=3, step=1)
        draw_parallel(int(n), title="Parallel Structure")

        st.write("请输入每个并联支路元器件在任务时刻的可靠度 R_i（0~1）：")
        cols = st.columns(4)
        R_list = []
        for i in range(int(n)):
            with cols[i % 4]:
                Ri = st.number_input(f"R{i + 1}", min_value=0.0, max_value=1.0, value=0.95, format="%.6f")
            R_list.append(Ri)

        R_sys = compute_parallel(R_list)

    else:
        # 混合：串联若干 stage，每个 stage 是 single 或 parallel group
        n_stage = st.number_input("Stage 数量（串联级数）", min_value=1, max_value=30, value=3, step=1)

        stages = []
        stage_Rs = []

        st.write("为每个 Stage 选择类型，并输入该 Stage 内的元器件可靠度：")
        for s in range(int(n_stage)):
            with st.expander(f"Stage {s + 1} 设置", expanded=(s == 0)):
                stg_type = st.selectbox(f"Stage {s + 1} 类型", ["单元件", "并联组"], key=f"stg_type_{s}")

                if stg_type == "单元件":
                    Ri = st.number_input(
                        f"Stage {s + 1} 元器件可靠度 R",
                        min_value=0.0, max_value=1.0, value=0.99, format="%.6f",
                        key=f"stg_single_R_{s}"
                    )
                    R_stage = compute_series([Ri])  # 就是Ri
                    stages.append({"type": "single", "n": 1})
                    stage_Rs.append(R_stage)

                else:
                    k = st.number_input(
                        f"Stage {s + 1} 并联支路元器件数量 k",
                        min_value=2, max_value=50, value=2, step=1,
                        key=f"stg_par_k_{s}"
                    )
                    cols = st.columns(4)
                    R_par = []
                    for j in range(int(k)):
                        with cols[j % 4]:
                            Rij = st.number_input(
                                f"R{s + 1}-{j + 1}",
                                min_value=0.0, max_value=1.0, value=0.95, format="%.6f",
                                key=f"stg_par_R_{s}_{j}"
                            )
                        R_par.append(Rij)

                    R_stage = compute_parallel(R_par)
                    stages.append({"type": "parallel", "n": int(k)})
                    stage_Rs.append(R_stage)

        draw_mixed(stages, title="Mixed Structure: Series of Stages (each stage can be parallel)")

        # 系统可靠度：串联各Stage
        R_sys = compute_series(stage_Rs)

    # -----------------------
    # 输出系统结果
    # -----------------------
    if R_sys is None:
        st.error("电路可靠度计算失败：请检查输入是否为 0~1 的数值。")
    else:
        F_sys = 1.0 - R_sys
        col1, col2, col3 = st.columns(3)
        col1.metric("电路可靠度 R_sys", f"{R_sys:.8f}")
        col2.metric("累计失效概率 F_sys", f"{F_sys:.8f}")
        col3.metric("评估时刻 t", f"{t_mission:.4f}")

        # 可选：把输入和中间结果展开给用户看（混合结构时尤其有用）
        # ✅ 公式用 LaTeX，确保渲染正确
        with st.expander("查看计算明细"):
            st.write("假设：元器件失效相互独立；输入的 $R_i$ 均为同一评估时刻 $t$ 的可靠度。")

            if model_type == "串联模型":
                st.latex(r"R_{\mathrm{sys}}(t)=\prod_{i=1}^{n} R_i(t)")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Component": [f"C{i + 1}" for i in range(len(R_list))],
                    "R_i(t)": R_list
                }))

            elif model_type == "并联模型":
                st.latex(r"R_{\mathrm{sys}}(t)=1-\prod_{i=1}^{n}\left(1-R_i(t)\right)")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Branch": [f"C{i + 1}" for i in range(len(R_list))],
                    "R_i(t)": R_list
                }))

            else:
                st.latex(r"R_{\mathrm{sys}}(t)=\prod_{s=1}^{S} R_{\mathrm{stage},s}(t)")
                st.latex(
                    r"R_{\mathrm{stage}}(t)=1-\prod_{j=1}^{k}\left(1-R_j(t)\right)\quad(\text{if stage is parallel})")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Stage": [f"Stage {i + 1}" for i in range(len(stage_Rs))],
                    "R_stage(t)": stage_Rs
                }))


