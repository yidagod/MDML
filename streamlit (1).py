#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations


# In[7]:


# 设置页面标题
st.title("中老年人心血管发病风险全周期预测系统")

# 上传数据文件
uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx"])
if uploaded_file is not None:
    # 读取数据
    if uploaded_file.name.endswith(".csv"):
        train_data = pd.read_csv(uploaded_file)
    else:
        train_data = pd.read_excel(uploaded_file)

    # 显示数据
    st.subheader("数据预览")
    st.write(train_data.head())

    # 定义变量
    T = train_data.iloc[:, 2:6]  # 假设第3-6列是处理变量
    Y = train_data.iloc[:, 0]    # 假设第1列是目标变量
    X = train_data.iloc[:, 6:]   # 假设第7列及以后是协变量

    # 将数据集分为训练集和测试集
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )

    # 训练 XGBoost 模型用于预测发病风险
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_xgb.fit(X_train, y_train)

    # 预测发病风险
    st.subheader("预测发病风险")
    # 选择一个用户输入的功能来调整协变量
    input_features = {}
    for col in X.columns:
        input_features[col] = st.number_input(f"输入 {col}", value=float(X[col].mean()))
    
    input_df = pd.DataFrame([input_features])
    predicted_risk = model_xgb.predict(input_df)[0]
    st.write(f"预测的发病风险: {predicted_risk:.4f}")

    # 当前风险预测解释
    st.write("当前风险是基于输入的协变量计算的。以下是各协变量的特征重要性：")
    fig, ax = plt.subplots()
    xgb.plot_importance(model_xgb, ax=ax)
    st.pyplot(fig)

    # 因果效应估计部分
    st.subheader("因果效应估计")
    # 定义 LinearDML 模型
    dml = LinearDML(model_y=RandomForestRegressor(), model_t=RandomForestRegressor())
    dml.fit(y_train, T_train, X=X_train.values)

    # 预测因果效应
    effects_dml = dml.effect(X_test.values)

    # 打印平均因果效应
    st.write("平均因果效应:", effects_dml.mean())

    # 可以修改处理变量，查看对发病风险的影响
    st.subheader("调整处理变量查看效果")
    treatment_vars = T.columns.tolist()
    for var in treatment_vars:
        new_value = st.number_input(f"调整 {var}", value=float(T[var].mean()), key=var)
        # 创建新的处理变量数据
        new_T = T_train.copy()
        new_T[var] = new_value
        # 重新拟合模型并预测效应（简化版，实际应用中可能需要重新训练模型）
        dml.fit(y_train, new_T, X=X_train.values)
        new_effects = dml.effect(X_test.values)
        st.write(f"调整 {var} 后的平均因果效应: {new_effects.mean():.4f}")

    # 绘制变量组合的因果效应图
    combination_effects = {}
    num_treatments = T.shape[1]
    for r in range(1, num_treatments + 1):
        for combination in combinations(range(num_treatments), r):
            combined_treatment = T_train.iloc[:, list(combination)].mean(axis=1)
            dml.fit(y_train, combined_treatment, X=X_train.values)
            effect = dml.effect(X_test.values)
            combination_effects[f"Combination {'+'.join(map(str, combination))}"] = effect.mean()

    effects_df = pd.DataFrame.from_dict(combination_effects, orient='index', columns=['Effect Value'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x=effects_df.index, y='Effect Value', data=effects_df, palette='viridis')
    plt.title('单独及组合处理变量的因果效应')
    plt.xlabel('处理组合')
    plt.ylabel('效应值')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # 使用 XGBoost 模型进行解释和特征重要性
    st.subheader("XGBoost 模型解释")
    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model_xgb)
    plt.title('XGBoost 特征重要性')
    st.pyplot(plt)


# In[2]:





# In[ ]:




