import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np

# ======================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ======================
df_raw = pd.read_csv('GlobalTemperatures_Optimized_Half2_English.csv')

# Преобразование даты и извлечение года/месяца
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw['Year'] = df_raw['Date'].dt.year
df_raw['Month'] = df_raw['Date'].dt.month
df_raw = df_raw.dropna(subset=['AverageTemperature'])

# Определение полушария по широте
def parse_latitude(lat_str):
    if pd.isna(lat_str):
        return np.nan
    val = float(lat_str[:-1])
    if lat_str.endswith('S'):
        val = -val
    return val

df_raw['Latitude_Deg'] = df_raw['Latitude'].apply(parse_latitude)
df_raw['Hemisphere'] = df_raw['Latitude_Deg'].apply(lambda x: 'North' if x >= 0 else 'South')

# Широтные зоны (для анализа по климатическим поясам)
def get_lat_band(lat):
    if pd.isna(lat):
        return 'Unknown'
    if lat >= 60:
        return 'Arctic (60°N+)'
    elif lat >= 30:
        return 'North Temperate (30°–60°N)'
    elif lat >= 0:
        return 'Tropics North (0°–30°N)'
    elif lat >= -30:
        return 'Tropics South (0°–30°S)'
    elif lat >= -60:
        return 'South Temperate (30°–60°S)'
    else:
        return 'Antarctic (60°S+)'

df_raw['Lat_Band'] = df_raw['Latitude_Deg'].apply(get_lat_band)

# ===== Агрегированные датасеты =====

# 1. Глобальный годовой тренд
df_global_yearly = df_raw.groupby('Year')['AverageTemperature'].mean().reset_index()

# 2. Глобальная сезонность (по месяцам)
df_global_monthly = df_raw.groupby('Month')['AverageTemperature'].mean().reset_index()
df_global_monthly['Month_Name'] = pd.to_datetime(df_global_monthly['Month'], format='%m').dt.month_name()

# 3. Средняя температура по странам (за весь период)
df_country_avg = df_raw.groupby('Country')['AverageTemperature'].mean().reset_index()
df_country_avg = df_country_avg.sort_values('AverageTemperature', ascending=False)

# 4. Температура по полушариям по годам
df_hemi_yearly = df_raw.groupby(['Year', 'Hemisphere'])['AverageTemperature'].mean().reset_index()

# 5. Тепловая карта: Год × Страна (только топ стран с достаточными данными)
country_counts = df_raw['Country'].value_counts()
top_countries = country_counts[country_counts >= 100].index
df_top = df_raw[df_raw['Country'].isin(top_countries)]
df_heatmap_country = df_top.groupby(['Year', 'Country'])['AverageTemperature'].mean().reset_index()

# 6. Температура по широтным зонам по годам
df_latband_yearly = df_raw.groupby(['Year', 'Lat_Band'])['AverageTemperature'].mean().reset_index()

# 7. Тепловая карта: Месяц × Широтная зона
df_heatmap_lat_month = df_raw.groupby(['Month', 'Lat_Band'])['AverageTemperature'].mean().reset_index()

# Уникальные страны и годы
countries = ['All'] + sorted(df_raw['Country'].unique())
years = sorted(df_raw['Year'].unique())
min_year, max_year = min(years), max(years)

# ======================
# ИНИЦИАЛИЗАЦИЯ DASH
# ======================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"]
)
server = app.server

# ======================
# МАКЕТ (LAYOUT)
# ======================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("🌍 Монитор воздействия на окружающую среду: Анализ глобальной температуры", 
                className="text-center my-4 fw-bold"),
        html.Div([
            dcc.Link("📊 Сырые данные и глобальные тренды", href="/", className="btn btn-outline-primary m-2"),
            dcc.Link("🔍 Расширенный анализ", href="/analysis", className="btn btn-outline-success m-2")
        ], className="text-center mb-4")
    ]),
    html.Div(id='page-content')
])

# Страница 1: Сырые данные и тренды
raw_layout = html.Div([
    html.H2("📊 Сырые данные и глобальные тренды", className="text-center mb-4"),

    # Фильтры
    html.Div([
        html.Div([
            html.Label("Страна:", className="form-label"),
            dcc.Dropdown(
                id='country-filter-raw',
                options=[{'label': c, 'value': c} for c in countries],
                value='All',
                className="form-control"
            )
        ], className="col-md-4"),
        html.Div([
            html.Label("Годы:", className="form-label"),
            dcc.RangeSlider(
                id='year-slider-raw',
                min=min_year,
                max=max_year,
                step=1,
                value=[max(1850, min_year), min(2010, max_year)],
                marks={y: str(y) for y in range(min_year, max_year+1, 20)},
                className="mt-2"
            )
        ], className="col-md-8")
    ], className="row mb-4"),

    # KPI-карточки
    html.Div(id='kpi-cards-raw', className="row mb-4"),

    # Таблица с первыми 20 строками данных
    html.H4("Первые 20 строк исходных данных", className="mt-4 mb-2"),
    dash_table.DataTable(
        data=df_raw.head(20).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_raw.columns if i != 'Latitude_Deg'],
        page_size=10,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '12px'}
    ),

    # Графики
    html.H3("Глобальный тренд температуры", className="mt-5"),
    dcc.Graph(id='global-trend-plot', className="mb-4"),

    html.H3("Сезонность температуры по месяцам", className="mt-5"),
    dcc.Graph(id='seasonality-plot', className="mb-4"),

    html.H3("Средняя температура по странам", className="mt-5"),
    dcc.Graph(id='country-bar-plot', className="mb-4"),

    html.H3("Температура по полушариям", className="mt-5"),
    dcc.Graph(id='hemisphere-plot', className="mb-4"),
])

# Страница 2: Расширенный анализ
analysis_layout = html.Div([
    html.H2("🔍 Расширенный анализ окружающей среды", className="text-center mb-4"),

    html.H3("Тепловая карта: Год × Страна (топ стран)", className="mt-4"),
    dcc.Graph(id='heatmap-country', figure=px.density_heatmap(
        df_heatmap_country, x='Year', y='Country', z='AverageTemperature',
        color_continuous_scale='RdYlBu_r', title="Средняя температура по годам и странам"
    ), className="mb-4"),

    html.H3("Температура по широтным зонам", className="mt-5"),
    dcc.Graph(id='latband-line', figure=px.line(
        df_latband_yearly, x='Year', y='AverageTemperature', color='Lat_Band',
        title="Температурные тренды по широтным зонам"
    ), className="mb-4"),

    html.H3("Тепловая карта: Месяц × Широтная зона", className="mt-5"),
    dcc.Graph(id='heatmap-lat-month', figure=px.density_heatmap(
        df_heatmap_lat_month, x='Month', y='Lat_Band', z='AverageTemperature',
        color_continuous_scale='RdYlBu_r',
        category_orders={"Lat_Band": [
            'Arctic (60°N+)', 'North Temperate (30°–60°N)', 'Tropics North (0°–30°N)',
            'Tropics South (0°–30°S)', 'South Temperate (30°–60°S)', 'Antarctic (60°S+)'
        ]},
        title="Средняя температура по месяцам и широтным зонам"
    ), className="mb-4"),

    html.Div([
        html.H4("Ключевые наблюдения", className="mt-5"),
        html.Ul([
            html.Li("Глобальная температура устойчиво растёт с XIX века."),
            html.Li("Сезонные колебания сильнее выражены в умеренных широтах."),
            html.Li("Полярные регионы нагреваются быстрее — признак полярного усиления."),
            html.Li("Тропики демонстрируют наименьшую годовую изменчивость.")
        ], className="alert alert-info")
    ])
])

# ======================
# CALLBACKS
# ======================

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/analysis':
        return analysis_layout
    return raw_layout

@app.callback(
    Output('kpi-cards-raw', 'children'),
    Output('global-trend-plot', 'figure'),
    Output('seasonality-plot', 'figure'),
    Output('country-bar-plot', 'figure'),
    Output('hemisphere-plot', 'figure'),
    Input('country-filter-raw', 'value'),
    Input('year-slider-raw', 'value')
)
def update_raw_page(country, year_range):
    # Фильтрация данных
    dff = df_raw.copy()
    dff = dff[(dff['Year'] >= year_range[0]) & (dff['Year'] <= year_range[1])]
    if country != 'All':
        dff = dff[dff['Country'] == country]

    # KPI
    kpi_cards = [
        html.Div(html.Div([
            html.H5("Записей", className="card-title"),
            html.H4(f"{len(dff):,}", className="card-text")
        ], className="card-body"), className="col-md-3"),
        html.Div(html.Div([
            html.H5("Ср. температура", className="card-title"),
            html.H4(f"{dff['AverageTemperature'].mean():.2f}°C", className="card-text")
        ], className="card-body"), className="col-md-3"),
        html.Div(html.Div([
            html.H5("Страны", className="card-title"),
            html.H4(f"{dff['Country'].nunique()}", className="card-text")
        ], className="card-body"), className="col-md-3"),
        html.Div(html.Div([
            html.H5("Годы", className="card-title"),
            html.H4(f"{dff['Year'].nunique()}", className="card-text")
        ], className="card-body"), className="col-md-3")
    ]

    # График 1: Глобальный тренд
    if country == 'All':
        trend_data = df_global_yearly[
            (df_global_yearly['Year'] >= year_range[0]) & 
            (df_global_yearly['Year'] <= year_range[1])
        ]
        fig_trend = px.line(trend_data, x='Year', y='AverageTemperature',
                            title="Глобальный тренд средней температуры")
    else:
        country_yearly = dff.groupby('Year')['AverageTemperature'].mean().reset_index()
        fig_trend = px.line(country_yearly, x='Year', y='AverageTemperature',
                            title=f"Температурный тренд: {country}")

    # График 2: Сезонность
    if country == 'All':
        fig_season = px.bar(df_global_monthly, x='Month', y='AverageTemperature',
                            title="Глобальная сезонность температур")
    else:
        monthly = dff.groupby('Month')['AverageTemperature'].mean().reset_index()
        monthly = monthly.merge(df_global_monthly[['Month', 'Month_Name']], on='Month')
        fig_season = px.bar(monthly, x='Month', y='AverageTemperature',
                            title=f"Сезонность в {country}")

    # График 3: Страны
    if country == 'All':
        top_countries_plot = df_country_avg.head(20)
        fig_country = px.bar(top_countries_plot, x='AverageTemperature', y='Country',
                             orientation='h', title="Топ-20 стран по средней температуре")
    else:
        fig_country = px.bar([{'Country': country, 'Avg': dff['AverageTemperature'].mean()}],
                             x='Avg', y='Country', orientation='h',
                             title=f"Средняя температура в {country}")

    # График 4: Полушария
    if country == 'All':
        hemi_data = df_hemi_yearly[
            (df_hemi_yearly['Year'] >= year_range[0]) & 
            (df_hemi_yearly['Year'] <= year_range[1])
        ]
        fig_hemi = px.line(hemi_data, x='Year', y='AverageTemperature', color='Hemisphere',
                           title="Сравнение температур: Север vs Юг")
    else:
        hemi_dff = dff.groupby(['Year', 'Hemisphere'])['AverageTemperature'].mean().reset_index()
        fig_hemi = px.line(hemi_dff, x='Year', y='AverageTemperature', color='Hemisphere',
                           title=f"Полушария в {country}")

    return kpi_cards, fig_trend, fig_season, fig_country, fig_hemi

# ======================
# ЗАПУСК
# ======================
if __name__ == '__main__':
    app.run_server(debug=True)
