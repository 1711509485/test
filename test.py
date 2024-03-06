import os

from datetime import datetime

import streamlit as st

import numpy as np
import pandas as pd
import warnings
import ast
import pickle
import tempfile
from io import BytesIO
import time
# from matplotlib import pyplot as plt
# warnings.filterwarnings("ignore")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams["axes.unicode_minus"]=False

def read_file(uploaded_file):
    # 获取上传文件的名称和扩展名
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    # 将上传的文件保存到临时文件中
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # 根据扩展名读取文件
    if file_extension == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_extension == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension in ('.xlsx', '.xls'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    # 删除临时文件
    os.remove(file_path)

    return data


# 定义阈值函数
def threshold_func(x):
    if pd.api.types.is_numeric_dtype(x.dtype):  # 如果数据类型是数字类型
        return [x.min(), x.max()]
    else:
        return [np.nan, np.nan]  # 对于非数字类型，设置为空值

def save_file_to_local_path_sta(file, local_path):
    file.to_csv(local_path, encoding='utf-8-sig')
    st.success(f"文件已保存到本地路径: {local_path}")

def build_dipin(data1,data2,data3,i):
    col1, col2, col3, col4, col5, col6 = st.columns(6)


    with col1:
        st.write(data1[i])
    with col2:
        # st.write(data2[i])
        selected_op = st.selectbox(f'原类型：{data2[i]}',options=options_list,key=f'geshi_{i}')
        if selected_op == 'int64':
            result_types.iloc[i,1] = 'int64'
        elif selected_op == 'float64':
            result_types.iloc[i,1] = 'float64'
        elif selected_op == 'string':
            result_types.iloc[i,1] = 'string'
        elif selected_op == 'object':
            result_types.iloc[i,1] = 'object'


    with col3:
        data3[i] = st.text_input('',key=f'yuzhi_{i}',value=data3[i])
    with col4:
        mean0_key = f'mean_{i}'
        mean0 = st.checkbox('Mean', key=mean0_key, value=True if xu_da_sta1.iloc[0, i] == 1 else False)
        if mean0:
            xu_da_sta1.iloc[0, i] = 1
            mode0_key = f'mode_{i}'
            # 如果 mode0 被选中，则取消勾选，并将 mode0 的状态设置为 False
            if st.session_state.get(mode0_key, False):
                st.session_state[mode0_key] = False
        else:
            xu_da_sta1.iloc[0, i] = 0  # 如果取消勾选 'Mean' 复选框，则将对应的值设置为0

    with col5:
        mode0_key = f'mode_{i}'
        mode0 = st.checkbox('Mode', key=mode0_key)
        if mode0:
            xu_da_sta1.iloc[0, i] = 2
            mean0_key = f'mean_{i}'
            # 如果 mean0 被选中，则取消勾选
            if st.session_state.get(mean0_key, False):
                st.session_state[mean0_key] = False


    with col6:
        yichang0 = st.checkbox('Yichang', key=f'yichang_{i}',value=True if xu_da_sta2.iloc[0, i] == 3 else False)
        if yichang0:
            xu_da_sta2.iloc[0, i] = 3



# 从字符串中提取最小值和最大值
def extract_min_max(cell):
    try:
        values = ast.literal_eval(cell)  # 使用ast.literal_eval将字符串转换为列表
        values = [x if not np.isnan(x) else None for x in values]  # 将nan替换为None
        return min(values), max(values)
    except (SyntaxError, ValueError):  # 处理无法解析的字符串或NaN
        return np.nan, np.nan


def convert_to_integer(x):
    try:
        if isinstance(x, str):
            if '.' in x:
                return int(float(x))  # 将数字字符串转为整数
            elif x.isdigit():
                return int(x)  # 将纯整数字符串转为整数
            else:
                return x  # 对于其他字符串，保持原样
        elif pd.notna(pd.to_numeric(x, errors='coerce')):  # 判断是否为浮点数
            return int(x)  # 将浮点数转为整数并四舍五入
    except ValueError:
        pass  # 忽略转换失败的情况，保持原样
    return x  # 其他情况保持原样

# 生成结果文件并提供下载链接
def download_result(data1,formatted_time):
    df = data1

    # 将 DataFrame 转换为 CSV 格式的字节流
    csv = df.to_csv(index=False).encode('utf-8-sig')

    # 将字节流转换为 BytesIO 对象
    csv_io = BytesIO(csv)

    # 提供下载链接
    st.download_button(label="下载文件至本地路径", data=csv_io, file_name='低频处理后的炉次_{}.csv'.format(
                                                                      formatted_time), mime='text/csv')

#低频数据筛选==============================================================
st.title("转炉质量根因分析模型离线仿真器")
st.subheader("2-低频数据筛选")

uploaded_file_sta = st.file_uploader("上传低频数据", type=['pkl','csv','xlsx'])  # ,'csv'
# 检查是否有文件上传
if uploaded_file_sta is not None:

    # 1、读取低频数据======================
    file = 'F:\数据处理\文件处理'
    # df_sta = pd.read_pickle(os.path.join(file,'ZI 低频-脱敏发出20230823.pkl'))
    df_sta = read_file(uploaded_file_sta)
    # 读取双渣操作的炉次
    ZI_shuangzha = pd.read_excel(os.path.join(file, '双渣炉次炉次号20230918.xlsx'), sheet_name='ZI')  # 读取低频数据
    # 2、获取低频数据和高频数据的中文表头
    df_sta_name = pd.read_excel(os.path.join(file, '低频数据和高频数据表头20230823.xlsx'), sheet_name='低频数据')  # 读取低频数据
    df_yuzhi = pd.read_excel(os.path.join(file, '各变量阈值上下限.xlsx'))
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    st.success("低频数据载入成功,载入{}炉次低频数据".format(df_sta.shape[0]))
    # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
    # df_sta.columns = df_sta_name.iloc[0, :]

    # 4、添加计算量
    fl_sec = ['预装', '前装', '吹炼前期', '吹炼中期', 'TSC之后', '后装']
    for i in range(6):
        df_sta['{}_辅料加入总量'.format(fl_sec[i])] = df_sta[['{}_辅原料{}加入量'.format(fl_sec[i], j) for j in range(1, 16)]].sum(
            axis=1)
    v_sta_name = df_sta.columns.tolist()

    data_sta1 = df_sta.copy()
    nan_tso = df_sta[df_sta['温度控制目标符合标签'].isna()].index.tolist()
    row_index = np.unique(nan_tso)
    data_sta1.drop(index=row_index, axis=0, inplace=True)
    data_sta1["TSO检测碳含量"] = pd.to_numeric(data_sta1["TSO检测碳含量"], errors="coerce")


    ZI_shuangzha_sum = []
    for i in range(ZI_shuangzha.shape[0]):
        ZI_shuangzha_name = data_sta1[data_sta1['标识列'] == ZI_shuangzha.iloc[i][0]].index.tolist()
        ZI_shuangzha_sum.extend(ZI_shuangzha_name)

    shuangzha_remove_da = df_sta.loc[ZI_shuangzha_sum]

    shuangzha_remove_da = shuangzha_remove_da.applymap(convert_to_integer)
    shuangzha_remove_da['TSO检测碳含量'] = shuangzha_remove_da['TSO检测碳含量'].astype(str)
    data_sta1.drop(index=np.array(ZI_shuangzha_sum), axis=0, inplace=True)
    st.success('去除双渣炉次{}，共有{}炉次低频数据含有温度标签'.format(shuangzha_remove_da.shape[0],data_sta1.shape[0]))


    #获取最小最大值（阈值）
    thresholds = data_sta1.agg(threshold_func)
    # 遍历theresholds的每一列
    for column in thresholds.columns:
        # 如果列名在table2的第一列中
        if column in df_yuzhi.iloc[:, 0].values:
            # 找到table2中对应的第二列的值
            new_value = df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 1]
            # 将table1中对应列的第一行修改为新值
            thresholds.at[0, column] = new_value
            # 将table1中对应列的第二行修改为新值
            if not pd.isnull(df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 2]):
                thresholds.at[1, column] = df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 2]

    thresholds_list = thresholds.apply(lambda x: f'[{x[0]}, {x[1]}]').tolist()

    cleaned_da = data_sta1[data_sta1['温度控制目标符合标签'] == 0]
    norclea_da = data_sta1[data_sta1['温度控制目标符合标签'] == 1]
    data_types = data_sta1.dtypes
    data_types_0 = []
    for col in data_sta1.columns:
        data_types_0.append({"Column Name": col, "Data Type": data_sta1[col].dtype})

    # 将结果放入 DataFrame
    result_types = pd.DataFrame(data_types_0)


    # 获取温度不命中的均值和众数
    numeric_columns_0 = cleaned_da.select_dtypes(exclude=['object'])
    mean_values_cle = numeric_columns_0.mean()
    mode_values_cle = numeric_columns_0.mode()
    mode_values_cle0 = mode_values_cle.iloc[0]

    # 获取温度命中的均值和众数
    numeric_columns_1 = norclea_da.select_dtypes(exclude=['object'])
    mean_values_nor = numeric_columns_1.mean()
    mode_values_nor = numeric_columns_1.mode()
    mode_values_nor0 = mode_values_nor.iloc[0]

    # 用均值填充
    cleaned_da0 = cleaned_da.fillna(mean_values_cle)
    norclea_da0 = norclea_da.fillna(mean_values_nor)
    data_sta_mean = pd.concat([cleaned_da0, norclea_da0], axis=0)
    data_sta_mean["TSO检测碳含量"] = pd.to_numeric(data_sta_mean["TSO检测碳含量"], errors="coerce")


    # 用众数填充
    cleaned_da1 = cleaned_da.fillna(mode_values_cle0)
    norclea_da1 = norclea_da.fillna(mode_values_nor0)
    data_sta_mode = pd.concat([cleaned_da1, norclea_da1], axis=0)
    data_sta_mode["TSO检测碳含量"] = pd.to_numeric(data_sta_mode["TSO检测碳含量"], errors="coerce")



    columns_da1 = data_sta1.columns
    columns_name_fe = ['铁水加入量','铁水温度','铁水C','铁水Si','铁水Mn','铁水P','铁水S','废钢量']
    # 创建一个字典，将每个列名映射到值为 1
    values_da = {col: 0 for col in columns_da1}
    # 创建一个新的 DataFrame
    xu_da_sta1 = pd.DataFrame(values_da, index=[0])
    xu_da_sta1.loc[xu_da_sta1.index[0], columns_name_fe] = 1
    xu_da_sta2 = pd.DataFrame(values_da, index=[0])
    xu_da_sta2.loc[xu_da_sta2.index[0], columns_name_fe] = 3

    if 'da_type'not in st.session_state:
        st.session_state.da_type = None
    options_list = ['int64','float64','string','object']

    col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(6)
    with col_1:
        st.write('列名')
    with col_2:
        st.write('数据类型')
    with col_3:
        st.write('阈值')
    with col_4:
        st.write('均值填充')
    with col_5:
        st.write('众数填充')
    with col_6:
        st.write('异常值剔除')

    for i in range(len(data_types)):
        mode_key = f'mode_{i}'
        st.session_state[mode_key] = st.session_state.get(mode_key, False)
        build_dipin(v_sta_name,data_types,thresholds_list,i)

    col_11,col_12,col_13,col_14,col_15=st.columns(5)
    with col_12:
        ensure_da = st.button('确认')
    with col_14:
        baocun = st.button('保存')

    # 定义结果的会话状态
    if 'result_me' not in st.session_state:
        st.session_state.result_me = None

    if 'result_mo' not in st.session_state:
        st.session_state.result_mo = None

    if 'result_yic' not in st.session_state:
        st.session_state.result_yic = None

    if 'ori' not in st.session_state:
        st.session_state.ori = None

    if 'result_di' not in st.session_state:
        st.session_state.result_di = None

    if 'out_nor' not in st.session_state:
        st.session_state.out_nor = False

    if 'outpath'not in st.session_state:
        st.session_state.outpath = None

    if 'indices_y' not in st.session_state:
        st.session_state.indices_y = None



    result_a = st.empty()
    result_b = st.empty()
    result_c = st.empty()
    result_d = st.empty()

    if ensure_da:
        indices_mo = xu_da_sta1.columns[xu_da_sta1.eq(2).any()]
        indices_me = xu_da_sta1.columns[xu_da_sta1.eq(1).any()]
        indices_yi = xu_da_sta2.columns[xu_da_sta2.eq(3).any()]
        st.session_state.indices_y = indices_yi
        indices_ori = xu_da_sta1.columns[xu_da_sta1.eq(0).any()]
        st.session_state.result_mo = data_sta_mode[indices_mo]
        st.session_state.result_me = data_sta_mean[indices_me]
        st.session_state.ori = data_sta1[indices_ori]
        result_di = pd.concat([data_sta_mean[indices_me], data_sta_mode[indices_mo], data_sta1[indices_ori]], axis=1)
        # 将列表转换为 DataFrame
        thresholds_list_df = pd.DataFrame(thresholds_list)
        # 获取最小值和最大值
        min_values, max_values = zip(*thresholds_list_df[0].apply(extract_min_max))
        # 创建新的DataFrame
        new_df = pd.DataFrame([min_values, max_values], columns=thresholds_list_df.index)
        new_df.columns = thresholds.columns
        # st.write(xu_da_sta1)
        # st.write(xu_da_sta2)
        # st.dataframe(result_types)
        # 遍历 result_types 表格中的每一行
        for index, row in result_types.iterrows():
            col_name = row["Column Name"]  # 获取列名
            data_type = row["Data Type"]  # 获取数据类型

            # 尝试将 data 表格中对应列的数据类型修改为 result_types 中指定的数据类型
            try:
                result_di[col_name] = result_di[col_name].astype(data_type)
            except ValueError:
                # 如果出现异常，保持原来的数据类型
                pass


        #清除异常炉次
        st.session_state.result_yic = None
        # 遍历每个列序列
        for ii in indices_yi:
            column_data = result_di.loc[:, ii]
            valid_indices = (column_data >= new_df.at[0, ii]) & (
                        column_data <= new_df.at[1, ii])
            invalid_indices = ~valid_indices  # 取反，获取不在范围内的行索引
            st.session_state.result_yic = pd.concat([st.session_state.result_yic, result_di[invalid_indices]])
            result_di = result_di[valid_indices]

        # 将剩余有效数据保存到 result_di 中
        st.session_state.result_di = result_di

    if baocun:
        if st.session_state.result_me is not None:
            with result_a:
                expander_mean = st.expander("均值填充结果：")
                expander_mean.dataframe(st.session_state.result_me)

        if st.session_state.result_mo is not None:
            with result_b:
                expander_mode = st.expander('众数填充结果：')
                expander_mode.dataframe(st.session_state.result_mo)

        if st.session_state.result_yic is not None:
            with result_c:
                expander_yic = st.expander('异常值剔除结果：剔除{}炉次'.format(st.session_state.result_yic.shape[0]))
                expander_yic.dataframe(st.session_state.result_yic[st.session_state.indices_y])

        if st.session_state.result_di is not None:
            with result_d:
                expander_di = st.expander("低频筛选后剩余炉次:{}".format(st.session_state.result_di.shape[0]))
                expander_di.dataframe(st.session_state.result_di)
                # out_nor = expander_di.button('保存文件', key='out_nor')
                # localpath = expander_di.text_input("保存路径（包括文件名.csv）:",
                #                                      os.path.join(file,
                #                                                   '低频筛选后的炉次_{}.csv'.format(
                #                                                       formatted_time
                #                                                   )
                #                                                   )
                #                                      )
                # st.session_state.outpath = localpath
        download_result(st.session_state.result_di,formatted_time)
    # if st.session_state.out_nor:
    #     save_file_to_local_path_sta(st.session_state.result_di, st.session_state.outpath)

else:
    # 如果没有文件上传，显示提示
    st.write("请上传文件")
