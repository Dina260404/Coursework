# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ========== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ==========
@st.cache_data
def load_data():
    # Прочитаем и очистим данные (как сделано ранее)
    with open('GlobalTemperatures_Optimized_Half2_fixed.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned_lines = []
    for line in lines:
        if line.startswith('"') and line.endswith('"\n'):
            line = line[1:-2] + '\n'
        elif line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        cleaned_lines.append(line)
    from io import StringIO
    csv_str = ''.join(cleaned_lines)
    df = pd.read_csv(StringIO(csv_str))
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    def parse_latlon(val):
        if 'N' in val: return float(val.replace('N', ''))
        elif 'S' in val: return -float(val.replace('S', ''))
        elif 'E' in val: return float(val.replace('E', ''))
        elif 'W' in val: return -float(val.replace('W', ''))
        else: return float(val)
    
    df['Latitude'] = df['Latitude'].apply(parse_latlon)
    df['Longitude'] = df['Longitude'].apply(parse_latlon)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

df = load_data()

# ========== ЗАГОЛОВОК ==========
st.set_page_config(layout="wide", page_title="🌍 Environmental Impact Monitor")
st.title("🌍 Environmental Impact Monitor: Global City Temperatures")

# ========== НАВИГАЦИЯ ==========
page = st.sidebar.radio("🧭 Navigation", ["Raw Data Visualization", "Analysis Results"])

# ========== СТРАНИЦА 1: RAW DATA ==========
if page == "Raw Data Visualization":
    st.header("📊 Raw Data Overview")

    # --- KPI ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Cities", df['City'].nunique())
    col3.metric("Countries", df['Country'].nunique())
    col4.metric("Years Covered", f"{df['Year'].min()} – {df['Year'].max()}")

    # --- ФИЛЬТРЫ ---
    st.sidebar.subheader("🔍 Filters")
    countries = st.sidebar.multiselect("Select Countries", options=sorted(df['Country'].unique()), default=[])
    years = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (1900, 2020))
    
    # Применяем фильтры
    filtered_df = df.copy()
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    filtered_df = filtered_df[(filtered_df['Year'] >= years[0]) & (filtered_df['Year'] <= years[1])]

    # --- ТАБЛИЦА ---
    st.subheader("📋 Sample Data (with sorting/search via built-in UI)")
    st.dataframe(filtered_df[['Date', 'City', 'Country', 'AverageTemperature', 'AverageTemperatureUncertainty']].head(20), use_container_width=True)

    # --- Распределения ---
    st.subheader("📈 Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.histogram(filtered_df, x='AverageTemperature', nbins=50, title="Temperature Distribution")
        st.plotly_chart(fig_temp, use_container_width=True)
    with col2:
        top_countries = filtered_df['Country'].value_counts().head(10)
        fig_country = px.bar(x=top_countries.index, y=top_countries.values, title="Top 10 Countries by Records")
        fig_country.update_layout(xaxis_title="Country", yaxis_title="Count")
        st.plotly_chart(fig_country, use_container_width=True)

    # --- Корреляция ---
    st.subheader("🌡️ Correlation Heatmap")
    numeric_cols = ['AverageTemperature', 'AverageTemperatureUncertainty', 'Latitude', 'Longitude', 'Year']
    corr = filtered_df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Scatter & Pie ---
    st.subheader("🔍 Additional Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(filtered_df, x='Longitude', y='Latitude', color='AverageTemperature',
                                 hover_data=['City', 'Country', 'Year'], title="Temperature by Location")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        pie_data = filtered_df['Country'].value_counts().head(6)
        fig_pie = px.pie(values=pie_data.values, names=pie_data.index, title="Country Share (Top 6)")
        st.plotly_chart(fig_pie, use_container_width=True)


# ========== СТРАНИЦА 2: ANALYSIS ==========
elif page == "Analysis Results":
    st.header("🔬 Temperature Trend & Clustering Analysis")

    # --- ФИЛЬТРЫ ---
    st.sidebar.subheader("🔍 Analysis Filters")
    countries = st.sidebar.multiselect("Countries", sorted(df['Country'].unique()), default=[])
    years = st.sidebar.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()), (1950, 2020))
    
    filtered_df = df.copy()
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    filtered_df = filtered_df[(filtered_df['Year'] >= years[0]) & (filtered_df['Year'] <= years[1])]

    # --- ВРЕМЕННОЙ РЯД (средняя температура по годам) ---
    st.subheader("📈 Global Temperature Trend")
    yearly = filtered_df.groupby('Year')['AverageTemperature'].mean().reset_index()
    
    # Линейная регрессия для тренда
    X = yearly[['Year']].values
    y = yearly['AverageTemperature'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=yearly['Year'], y=yearly['AverageTemperature'],
                                   mode='markers', name='Avg Temperature', opacity=0.7))
    fig_trend.add_trace(go.Scatter(x=yearly['Year'], y=y_pred, mode='lines', name=f'Trend (R² = {r2:.2f})', line=dict(color='red')))
    fig_trend.update_layout(title="Annual Average Temperature Trend", xaxis_title="Year", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- КЛАСТЕРИЗАЦИЯ ГОРОДОВ ---
    st.subheader("📍 City Clustering by Climate")
    city_avg = filtered_df.groupby(['City', 'Country', 'Latitude', 'Longitude'])['AverageTemperature'].mean().reset_index()
    
    if len(city_avg) >= 3:
        # Стандартизация и кластеризация
        features = city_avg[['Latitude', 'Longitude', 'AverageTemperature']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=min(5, len(city_avg)), random_state=42)
        city_avg['Cluster'] = kmeans.fit_predict(features_scaled)
        
        # Визуализация
        fig_clusters = px.scatter_mapbox(
            city_avg,
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            size='AverageTemperature',
            hover_name='City',
            hover_data=['Country', 'AverageTemperature'],
            zoom=1,
            title="City Clusters by Avg Temperature & Location"
        )
        fig_clusters.update_layout(mapbox_style="open-street-map", height=500)
        st.plotly_chart(fig_clusters, use_container_width=True)

        # --- KPI по кластерам ---
        st.subheader("📊 Cluster Insights")
        cluster_stats = city_avg.groupby('Cluster')['AverageTemperature'].agg(['mean', 'count']).round(2)
        st.dataframe(cluster_stats.rename(columns={'mean': 'Avg Temp', 'count': 'Cities'}), use_container_width=True)
        
        # Интерпретация
        hottest_cluster = cluster_stats['mean'].idxmax()
        hottest_temp = cluster_stats.loc[hottest_cluster, 'mean']
        st.info(f"🔥 Cluster {hottest_cluster} is the warmest (avg {hottest_temp}°C).")

    # --- Feature Importance (условная) ---
    st.subheader("🔍 Feature Influence on Temperature")
    corr_temp = filtered_df[['AverageTemperature', 'Latitude', 'Longitude', 'Year']].corr()['AverageTemperature'].drop('AverageTemperature')
    fig_imp = px.bar(x=corr_temp.index, y=corr_temp.values, title="Correlation with Temperature")
    fig_imp.update_layout(yaxis_title="Correlation Coefficient")
    st.plotly_chart(fig_imp, use_container_width=True)

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.write("💡 **Instructions to Run**:")
st.sidebar.code("pip install streamlit pandas plotly scikit-learn\nstreamlit run app.py")
st.sidebar.write("🌐 **Deploy**: Push to GitHub & deploy on [Streamlit Cloud](https://streamlit.io/cloud)")
