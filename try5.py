# -*- coding: utf-8 -*-
"""
随机森林建模与交互面板 - 韧性指数分析（预计算缓存版）
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import streamlit as st
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== 缓存文件路径（与恢复指数区分） ====================
IMPORTANCE_CACHE = "importance_resilience_cache.csv"
METRICS_CACHE = "metrics_resilience_cache.csv"
INDUSTRY_CORR_CACHE = "industry_corr_resilience_cache.csv"
PROVINCE_CORR_CACHE = "province_corr_resilience_cache.csv"
DATA_CACHE = "data_resilience_cached.parquet"


# ==================== 1. 数据加载与预处理 ====================
@st.cache_data
def load_and_preprocess():
    df = pd.read_excel('resilience_final.xlsx', sheet_name='Sheet1')

    # 提取年月
    if '年月' in df.columns:
        df['year'] = df['年月'] // 100
        df['month'] = df['年月'] % 100

    # 选择特征列
    feature_cols = [
        '对美国进口比例', '对美国出口比例', '对美国贸易比例',
        'GDP', 'GDP_pc', 'Population', 'Elderly', 'Third',
        'year', 'month'
    ]
    target_col = '韧性指数'
    industry_col = '商品编码'
    province_col = '注册地编码'

    # 删除缺失值
    df = df.dropna(subset=[target_col])
    df = df.dropna(subset=feature_cols)

    df[industry_col] = df[industry_col].astype(str)
    df[province_col] = df[province_col].astype(str)

    X = df[feature_cols + [industry_col, province_col]]
    y = df[target_col]

    # 预处理器
    categorical_features = [industry_col, province_col]
    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return X, y, preprocessor, df, numeric_features


# ==================== 2. 建模与分析（300次，并保存结果） ====================
def run_analysis_and_save(n_iter=300):
    """运行300次建模，保存所有结果到缓存文件"""
    X, y, preprocessor, df_raw, numeric_features = load_and_preprocess()

    feature_names = None
    all_importances = []
    all_r2 = []
    all_rmse = []

    industries = X['商品编码'].unique()
    provinces = X['注册地编码'].unique()

    industry_corr = {}
    province_corr = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(n_iter):
        progress_bar.progress((i + 1) / n_iter)
        status_text.text(f"正在建模：第 {i+1}/{n_iter} 次...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=-1))
        ])
        pipe.fit(X_train, y_train)

        if feature_names is None:
            X_enc = pipe.named_steps['preprocessor'].fit_transform(X_train)
            cat_feature_names = pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
                ['商品编码', '注册地编码'])
            feature_names = np.concatenate([numeric_features, cat_feature_names])

        importances = pipe.named_steps['regressor'].feature_importances_
        all_importances.append(importances)

        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        all_r2.append(r2)
        all_rmse.append(rmse)

        # 计算每个行业/省份的相关系数
        for ind in industries:
            sub = X_test[X_test['商品编码'] == ind].copy()
            if len(sub) > 5:
                corr = sub['对美国贸易比例'].corr(y_test.loc[sub.index])
                industry_corr.setdefault(ind, []).append(corr)
        for prov in provinces:
            sub = X_test[X_test['注册地编码'] == prov].copy()
            if len(sub) > 5:
                corr = sub['对美国贸易比例'].corr(y_test.loc[sub.index])
                province_corr.setdefault(prov, []).append(corr)

    progress_bar.empty()
    status_text.empty()

    # 平均重要性
    avg_importances = np.mean(all_importances, axis=0)
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': avg_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    # 性能指标
    avg_r2 = np.mean(all_r2)
    avg_rmse = np.mean(all_rmse)
    std_r2 = np.std(all_r2)
    std_rmse = np.std(all_rmse)

    # 行业/省份相关系数
    industry_corr_avg = {k: np.mean(v) for k, v in industry_corr.items() if v}
    industry_corr_df = pd.DataFrame(list(industry_corr_avg.items()), columns=['行业', 'correlation'])
    industry_corr_df = industry_corr_df.sort_values('correlation', ascending=False)

    province_corr_avg = {k: np.mean(v) for k, v in province_corr.items() if v}
    province_corr_df = pd.DataFrame(list(province_corr_avg.items()), columns=['省份', 'correlation'])
    province_corr_df = province_corr_df.sort_values('correlation', ascending=False)

    # 保存到 CSV
    importance_df.to_csv(IMPORTANCE_CACHE, index=False)
    pd.DataFrame([{
        'avg_r2': avg_r2,
        'std_r2': std_r2,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse
    }]).to_csv(METRICS_CACHE, index=False)
    industry_corr_df.to_csv(INDUSTRY_CORR_CACHE, index=False)
    province_corr_df.to_csv(PROVINCE_CORR_CACHE, index=False)

    # 保存预处理后的完整数据
    df_raw.to_parquet(DATA_CACHE, index=False)

    print("韧性指数分析：所有缓存文件已保存完成。")
    return importance_df, (avg_r2, std_r2, avg_rmse, std_rmse), industry_corr_df, province_corr_df, df_raw


def load_cached_results():
    """从缓存文件加载结果"""
    importance_df = pd.read_csv(IMPORTANCE_CACHE)
    metrics_df = pd.read_csv(METRICS_CACHE)
    avg_r2 = metrics_df['avg_r2'].iloc[0]
    std_r2 = metrics_df['std_r2'].iloc[0]
    avg_rmse = metrics_df['avg_rmse'].iloc[0]
    std_rmse = metrics_df['std_rmse'].iloc[0]
    industry_corr_df = pd.read_csv(INDUSTRY_CORR_CACHE)
    province_corr_df = pd.read_csv(PROVINCE_CORR_CACHE)
    df_raw = pd.read_parquet(DATA_CACHE)

    print("韧性指数分析：从缓存文件加载结果成功。")
    return importance_df, (avg_r2, std_r2, avg_rmse, std_rmse), industry_corr_df, province_corr_df, df_raw


# ==================== 3. Streamlit 交互面板 ====================
def main():
    st.set_page_config(layout="wide", page_title="韧性指数影响因素分析")
    st.title("分省分行业韧性指数影响因素分析")
    st.markdown("基于随机森林（300次重复随机划分）分析韧性指数的影响因素，并识别受美国影响最大的行业与省份。")

    # 检查是否存在所有缓存文件
    cache_exists = all(os.path.exists(f) for f in [
        IMPORTANCE_CACHE, METRICS_CACHE, INDUSTRY_CORR_CACHE, PROVINCE_CORR_CACHE, DATA_CACHE
    ])

    if not cache_exists:
        st.warning("未检测到预计算结果，正在运行300次随机森林建模（约6小时），请耐心等待...")
        with st.spinner("建模中..."):
            (importance_df, perf, industry_corr_df, province_corr_df, df_raw) = run_analysis_and_save(n_iter=300)
        st.success("建模完成！结果已缓存，后续启动将直接加载。")
        st.session_state.analyzed = True
        st.session_state.importance_df = importance_df
        st.session_state.perf = perf
        st.session_state.industry_corr_df = industry_corr_df
        st.session_state.province_corr_df = province_corr_df
        st.session_state.df_raw = df_raw
    else:
        # 直接加载缓存
        with st.spinner("加载预计算结果中..."):
            (importance_df, perf, industry_corr_df, province_corr_df, df_raw) = load_cached_results()
        st.session_state.analyzed = True
        st.session_state.importance_df = importance_df
        st.session_state.perf = perf
        st.session_state.industry_corr_df = industry_corr_df
        st.session_state.province_corr_df = province_corr_df
        st.session_state.df_raw = df_raw
        st.info("已加载预计算结果，无需重新建模。")

    # 展示结果
    if st.session_state.get('analyzed', False):
        importance_df = st.session_state.importance_df
        (avg_r2, std_r2, avg_rmse, std_rmse) = st.session_state.perf
        industry_corr_df = st.session_state.industry_corr_df
        province_corr_df = st.session_state.province_corr_df
        df_raw = st.session_state.df_raw

        st.subheader("模型性能（300次重复）")
        col1, col2 = st.columns(2)
        col1.metric("平均 R²", f"{avg_r2:.3f}", delta=f"±{std_r2:.3f}")
        col2.metric("平均 RMSE", f"{avg_rmse:.3f}", delta=f"±{std_rmse:.3f}")

        st.subheader("整体影响因素重要性")
        fig_imp = px.bar(importance_df.head(20), x='importance', y='feature', orientation='h',
                         title="Top 20 特征重要性",
                         labels={'importance': '平均重要性', 'feature': '特征'})
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("受美国影响最大的行业")
        fig_ind = px.bar(industry_corr_df.head(10), x='correlation', y='行业', orientation='h',
                         title="美国贸易比例与韧性指数相关系数（前10）",
                         labels={'correlation': '相关系数', '行业': '行业（商品编码）'})
        st.plotly_chart(fig_ind, use_container_width=True)

        st.subheader("受美国影响最大的省份")
        fig_prov = px.bar(province_corr_df.head(10), x='correlation', y='省份', orientation='h',
                          title="美国贸易比例与韧性指数相关系数（前10）",
                          labels={'correlation': '相关系数', '省份': '省份（注册地编码）'})
        st.plotly_chart(fig_prov, use_container_width=True)

        st.subheader("具体行业/省份关系探索")
        col1, col2 = st.columns(2)
        with col1:
            selected_industry = st.selectbox("选择行业（商品编码）", options=industry_corr_df['行业'].tolist())
            industry_data = df_raw[df_raw['商品编码'].astype(str) == selected_industry]
            if not industry_data.empty:
                fig_scatter = px.scatter(industry_data, x='对美国贸易比例', y='韧性指数',
                                         title=f"行业 {selected_industry} 美国贸易比例 vs 韧性指数",
                                         trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            selected_province = st.selectbox("选择省份（注册地编码）", options=province_corr_df['省份'].tolist())
            province_data = df_raw[df_raw['注册地编码'].astype(str) == selected_province]
            if not province_data.empty:
                fig_scatter2 = px.scatter(province_data, x='对美国贸易比例', y='韧性指数',
                                          title=f"省份 {selected_province} 美国贸易比例 vs 韧性指数",
                                          trendline="ols")
                st.plotly_chart(fig_scatter2, use_container_width=True)

        st.subheader("数据预览")
        st.dataframe(df_raw.head(100))
    else:
        st.info("等待分析结果...")


if __name__ == "__main__":
    main()