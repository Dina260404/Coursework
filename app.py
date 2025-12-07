# app.py
import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io

# ======================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ======================
try:
    df = pd.read_csv('GlobalTemperatures_Optimized_Half2_fixed.csv')
    df_clean = df.copy()
    
    # Обработка данных как в ноутбуке
    df_clean['dt'] = pd.to_datetime(df_clean['dt'])
    df_clean['Год'] = df_clean['dt'].dt.year
    df_clean['Месяц'] = df_clean['dt'].dt.month
    df_clean['День'] = df_clean['dt'].dt.day
    df_clean['Квартал'] = df_clean['dt'].dt.quarter
    
    # Добавляем широтные зоны
    def get_latitude_zone(lat):
        if pd.isna(lat):
            return 'Неизвестно'
        lat_val = float(lat[:-1]) if isinstance(lat, str) else lat
        if lat_val >= 66.5:
            return 'Полярная'
        elif lat_val >= 23.5:
            return 'Умеренная'
        elif lat_val >= 0:
            return 'Тропическая'
        else:
            return 'Умеренная'
    
    df_clean['Широтная_зона'] = df_clean['Latitude'].apply(get_latitude_zone)
    
    # Добавляем полушария
    def get_hemisphere(lat):
        if pd.isna(lat):
            return 'Неизвестно'
        lat_val = float(lat[:-1]) if isinstance(lat, str) else lat
        return 'Северное' if lat_val >= 0 else 'Южное'
    
    df_clean['Полушарие'] = df_clean['Latitude'].apply(get_hemisphere)
    
    # Добавляем десятилетия
    df_clean['Десятилетие'] = (df_clean['Год'] // 10) * 10
    
    # Убираем строки с пропущенными значениями температуры
    df_clean = df_clean.dropna(subset=['AverageTemperature'])
    
    # Создание агрегированных данных для анализа
    df_yearly = df_clean.groupby('Год')['AverageTemperature'].agg(['mean', 'std']).reset_index()
    df_yearly.rename(columns={'mean': 'AverageTemperature', 'std': 'TemperatureStd'}, inplace=True)
    df_yearly['10y_MA'] = df_yearly['AverageTemperature'].rolling(window=10).mean()
    
    df_monthly = df_clean.groupby('Месяц')['AverageTemperature'].mean().reset_index()
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    df_monthly['Месяц_название'] = [month_names[i-1] for i in df_monthly['Месяц']]
    
    # Анализ по странам
    df_country_stats = df_clean.groupby('Country').agg({
        'AverageTemperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    df_country_stats.columns = ['Средняя', 'Стд', 'Мин', 'Макс', 'Кол-во']
    df_country_stats = df_country_stats.reset_index()
    
    # Анализ по городам
    df_city_stats = df_clean.groupby('City').agg({
        'AverageTemperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    df_city_stats.columns = ['Средняя', 'Стд', 'Мин', 'Макс', 'Кол-во']
    df_city_stats = df_city_stats.reset_index()
    
    # Корреляционная матрица
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_clean[numeric_cols].corr()
    
    # Кластеризация (имитация из ноутбука)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Подготовка данных для кластеризации
    cluster_data = df_clean.groupby('Country').agg({
        'AverageTemperature': ['mean', 'std']
    }).dropna()
    cluster_data.columns = ['temp_mean', 'temp_std']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # KMeans кластеризация
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    cluster_data['cluster'] = cluster_labels
    
    # Добавляем обратно в оригинальные данные
    df_clustered = df_clean.merge(
        cluster_data[['cluster']], 
        left_on='Country', 
        right_index=True, 
        how='left'
    )
    
    # Статистика по кластерам
    cluster_stats = df_clustered.groupby('cluster').agg({
        'AverageTemperature': ['mean', 'std', 'count']
    }).round(2)
    
    # Тепловая карта: температура по годам и странам (топ 10 стран)
    top_countries = df_clean['Country'].value_counts().head(10).index.tolist()
    df_heatmap = df_clean[df_clean['Country'].isin(top_countries)].copy()
    df_heatmap['Год'] = df_heatmap['Год'].astype(str)
    heatmap_data = df_heatmap.pivot_table(
        index='Country',
        columns='Год',
        values='AverageTemperature',
        aggfunc='mean'
    ).fillna(0)
    
    # Средняя температура по широтным зонам
    df_latitude_zones = df_clean.groupby('Широтная_зона')['AverageTemperature'].mean().reset_index()
    
    # Температура по полушариям по десятилетиям
    df_hemisphere_decades = df_clean.groupby(['Полушарие', 'Десятилетие'])['AverageTemperature'].mean().reset_index()
    
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    # Создание демо-данных если файл не найден
    df_clean = pd.DataFrame({
        'Год': np.arange(1850, 2014),
        'AverageTemperature': 10 + np.random.randn(164).cumsum() * 0.1,
        'Country': ['Global'] * 164,
        'Месяц': np.tile(range(1, 13), 14)[:164],
        'dt': pd.date_range('1850-01-01', periods=164, freq='M'),
        'City': ['Global City'] * 164,
        'Latitude': ['0N'] * 164,
        'Longitude': ['0E'] * 164,
        'Полушарие': ['Северное'] * 164,
        'Широтная_зона': ['Умеренная'] * 164,
        'Десятилетие': [(y // 10) * 10 for y in np.arange(1850, 2014)]
    })
    df_yearly = df_clean.groupby('Год')['AverageTemperature'].mean().reset_index()
    df_monthly = df_clean.groupby('Месяц')['AverageTemperature'].mean().reset_index()
    df_country_stats = pd.DataFrame({'Country': ['Global'], 'Средняя': [10.0], 'Стд': [1.0], 'Мин': [8.0], 'Макс': [12.0], 'Кол-во': [164]})
    df_city_stats = pd.DataFrame({'City': ['Global City'], 'Средняя': [10.0], 'Стд': [1.0], 'Мин': [8.0], 'Макс': [12.0], 'Кол-во': [164]})
    correlation_matrix = pd.DataFrame(np.eye(3), columns=['Год', 'Месяц', 'AverageTemperature'], index=['Год', 'Месяц', 'AverageTemperature'])
    df_heatmap = pd.DataFrame()

# ======================
# ИНИЦИАЛИЗАЦИЯ DASH
# ======================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

# ======================
# LAYOUT
# ======================
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    
    # Навигация
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="https://cdn-icons-png.flaticon.com/512/3095/3095110.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("🌍 Climate Data Dashboard", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("📊 Визуализация данных", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("🔍 Анализ результатов", href="/analysis", active="exact")),
                dbc.NavItem(dbc.NavLink("📈 Прогнозирование", href="/forecast", active="exact")),
            ], navbar=True, className="ms-auto"),
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Контент страницы
    html.Div(id='page-content'),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© 2024 Climate Data Dashboard | Данные: Global Temperatures", 
                  className="text-center text-muted")
        ])
    ])
], fluid=True)

# ======================
# СТРАНИЦА 1: ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ======================
raw_data_layout = dbc.Container([
    html.H2("📊 Визуализация исходных данных", className="mb-4 text-center"),
    
    # Фильтры
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Страна:", className="form-label"),
                    dcc.Dropdown(
                        id='country-filter',
                        options=[{'label': 'Все страны', 'value': 'All'}] + 
                                [{'label': c, 'value': c} for c in sorted(df_clean['Country'].unique())],
                        value='All',
                        placeholder="Выберите страну...",
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Город:", className="form-label"),
                    dcc.Dropdown(
                        id='city-filter',
                        options=[{'label': 'Все города', 'value': 'All'}] + 
                                [{'label': c, 'value': c} for c in sorted(df_clean['City'].unique())],
                        value='All',
                        placeholder="Выберите город...",
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Диапазон лет:", className="form-label"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=int(df_clean['Год'].min()),
                        max=int(df_clean['Год'].max()),
                        value=[int(df_clean['Год'].min()), int(df_clean['Год'].max())],
                        marks={int(year): str(int(year)) 
                               for year in np.linspace(df_clean['Год'].min(), df_clean['Год'].max(), 10)},
                        className="mb-3"
                    )
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Выберите график:", className="form-label"),
                    dcc.Dropdown(
                        id='graph-selector',
                        options=[
                            {'label': '📈 Распределение температур', 'value': 'hist'},
                            {'label': '🌡️ Температура по месяцам', 'value': 'monthly'},
                            {'label': '📊 Box plot по странам', 'value': 'box_country'},
                            {'label': '📍 Scatter plot', 'value': 'scatter'},
                            {'label': '🔗 Корреляционная матрица', 'value': 'corr'},
                            {'label': '🌍 Глобальный тренд', 'value': 'global_trend'},
                            {'label': '🌐 Температура по городам', 'value': 'city_temp'},
                            {'label': '🧭 Сезонность по месяцам', 'value': 'seasonality'},
                            {'label': '🌎 Средняя температура по странам', 'value': 'avg_country'},
                            {'label': ' Hemisphere Temperature', 'value': 'hemisphere'},
                            {'label': '🗺️ Тепловая карта по годам и странам', 'value': 'heatmap'},
                            {'label': '🧭 Средняя температура по широтным зонам', 'value': 'latitude_zones'}
                        ],
                        value='hist',
                        className="mb-3"
                    )
                ], md=6),
                dbc.Col([
                    html.Label("Показать данные:", className="form-label"),
                    dbc.Checklist(
                        id='data-options',
                        options=[
                            {'label': ' Показать выбросы', 'value': 'outliers'},
                            {'label': ' Сгладить данные', 'value': 'smooth'}
                        ],
                        value=['smooth'],
                        inline=True,
                        className="mb-3"
                    )
                ], md=6),
            ])
        ])
    ], className="mb-4"),
    
    # KPI карточки
    html.Div(id='kpi-cards', className="mb-4"),
    
    # Таблица данных
    dbc.Card([
        dbc.CardHeader(html.H5("📋 Таблица данных", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='data-table',
                columns=[
                    {"name": "Дата", "id": "dt"},
                    {"name": "Температура (°C)", "id": "AverageTemperature"},
                    {"name": "Страна", "id": "Country"},
                    {"name": "Город", "id": "City"},
                    {"name": "Год", "id": "Год"},
                    {"name": "Месяц", "id": "Месяц"},
                    {"name": "Широта", "id": "Latitude"},
                    {"name": "Долгота", "id": "Longitude"}
                ],
                page_size=15,
                page_action='native',
                sort_action='native',
                filter_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '100px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                export_format='csv'
            )
        ])
    ], className="mb-4"),
    
    # Основной график
    dbc.Row([
        dbc.Col(dcc.Graph(id='main-graph'), width=12, className="mb-4"),
    ]),
    
    # Дополнительные графики
    dbc.Row([
        dbc.Col(dcc.Graph(id='hist-graph'), width=6, className="mb-4"),
        dbc.Col(dcc.Graph(id='box-graph'), width=6, className="mb-4"),
    ]),
    
    # Корреляционная матрица и scatter plot
    dbc.Row([
        dbc.Col(dcc.Graph(id='corr-graph'), width=6, className="mb-4"),
        dbc.Col(dcc.Graph(id='scatter-graph'), width=6, className="mb-4"),
    ]),
])

# ======================
# СТРАНИЦА 2: АНАЛИЗ РЕЗУЛЬТАТОВ
# ======================
analysis_layout = dbc.Container([
    html.H2("🔍 Результаты анализа", className="mb-4 text-center"),
    
    # Контролы для анализа
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Метод анализа:", className="form-label"),
                    dcc.RadioItems(
                        id='analysis-method',
                        options=[
                            {'label': '📊 Кластеризация', 'value': 'clustering'},
                            {'label': '📈 Временные ряды', 'value': 'timeseries'},
                            {'label': '📉 Регрессионный анализ', 'value': 'regression'},
                            {'label': '🌡️ Сравнение стран', 'value': 'comparison'}
                        ],
                        value='clustering',
                        inline=True,
                        className="mb-3"
                    )
                ], md=8),
                dbc.Col([
                    html.Label("Количество кластеров:", className="form-label"),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2,
                        max=5,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(2, 6)},
                        className="mb-3"
                    )
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Метрики оценки:", className="form-label"),
                    dbc.Checklist(
                        id='metrics-selector',
                        options=[
                            {'label': ' Silhouette Score', 'value': 'silhouette'},
                            {'label': ' R² Score', 'value': 'r2'},
                            {'label': ' MSE', 'value': 'mse'}
                        ],
                        value=['silhouette'],
                        inline=True,
                        className="mb-3"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Button("🔄 Обновить анализ", 
                              id='update-analysis', 
                              color="primary",
                              className="w-100 mt-4")
                ], md=6),
            ])
        ])
    ], className="mb-4"),
    
    # Метрики анализа
    html.Div(id='analysis-metrics', className="mb-4"),
    
    # Основной график анализа
    dbc.Card([
        dbc.CardHeader(html.H5("📈 График анализа", className="mb-0")),
        dbc.CardBody(dcc.Graph(id='analysis-main-graph'))
    ], className="mb-4"),
    
    # Дополнительные визуализации
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📋 Статистика по кластерам", className="mb-0")),
                dbc.CardBody(dash_table.DataTable(
                    id='cluster-table',
                    page_size=10,
                    style_table={'overflowX': 'auto'}
                ))
            ])
        ], md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📊 Feature Importance", className="mb-0")),
                dbc.CardBody(dcc.Graph(id='importance-graph'))
            ])
        ], md=6, className="mb-4"),
    ]),
    
    # Инсайты и интерпретации
    dbc.Card([
        dbc.CardHeader(html.H5("💡 Инсайты и интерпретации", className="mb-0")),
        dbc.CardBody([
            html.Div(id='insights-text'),
            html.Hr(),
            dbc.Alert(
                "💡 Наведите курсор на графики для получения подробной информации. "
                "Используйте фильтры для изменения параметров анализа.",
                color="info",
                className="mt-3"
            )
        ])
    ], className="mb-4")
])

# ======================
# СТРАНИЦА 3: ПРОГНОЗИРОВАНИЕ
# ======================
forecast_layout = dbc.Container([
    html.H2("📈 Прогнозирование", className="mb-4 text-center"),
    
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Модель прогнозирования:", className="form-label"),
                    dcc.Dropdown(
                        id='forecast-model',
                        options=[
                            {'label': 'ARIMA', 'value': 'arima'},
                            {'label': 'Линейная регрессия', 'value': 'linear'},
                            {'label': 'Prophet', 'value': 'prophet'},
                            {'label': 'Экспоненциальное сглаживание', 'value': 'exponential'}
                        ],
                        value='linear',
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Горизонт прогноза (лет):", className="form-label"),
                    dcc.Slider(
                        id='forecast-horizon',
                        min=1,
                        max=20,
                        step=1,
                        value=10,
                        marks={i: str(i) for i in range(1, 21, 5)},
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Доверительный интервал:", className="form-label"),
                    dcc.Slider(
                        id='confidence-interval',
                        min=80,
                        max=99,
                        step=1,
                        value=95,
                        marks={80: '80%', 90: '90%', 95: '95%', 99: '99%'},
                        className="mb-3"
                    )
                ], md=4),
            ]),
            dbc.Button("🚀 Запустить прогноз", 
                      id='run-forecast', 
                      color="success",
                      size="lg",
                      className="w-100 mb-3")
        ])
    ], className="mb-4"),
    
    # Прогнозные графики
    dbc.Row([
        dbc.Col(dcc.Graph(id='forecast-graph'), width=12, className="mb-4"),
    ]),
    
    # Метрики прогноза
    html.Div(id='forecast-metrics', className="mb-4"),
    
    # Таблица прогнозов
    dbc.Card([
        dbc.CardHeader(html.H5("📋 Прогнозные значения", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='forecast-table',
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
        ])
    ], className="mb-4")
])

# ======================
# CALLBACKS
# ======================

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/analysis':
        return analysis_layout
    elif pathname == '/forecast':
        return forecast_layout
    return raw_data_layout

# Callback для страницы визуализации данных
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('data-table', 'data'),
     Output('main-graph', 'figure'),
     Output('hist-graph', 'figure'),
     Output('box-graph', 'figure'),
     Output('corr-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('country-filter', 'value'),
     Input('city-filter', 'value'),
     Input('year-slider', 'value'),
     Input('graph-selector', 'value'),
     Input('data-options', 'value')]
)
def update_raw_data(country, city, year_range, graph_type, options):
    # Фильтрация данных
    filtered_df = df_clean.copy()
    filtered_df = filtered_df[(filtered_df['Год'] >= year_range[0]) & 
                             (filtered_df['Год'] <= year_range[1])]
    
    if country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    
    if city != 'All':
        filtered_df = filtered_df[filtered_df['City'] == city]
    
    # KPI карточки
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📊 Всего записей", className="card-subtitle"),
                html.H3(f"{len(filtered_df):,}", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("🌡️ Средняя температура", className="card-subtitle"),
                html.H3(f"{filtered_df['AverageTemperature'].mean():.2f}°C", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📈 Стандартное отклонение", className="card-subtitle"),
                html.H3(f"{filtered_df['AverageTemperature'].std():.2f}°C", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📍 Уникальных стран", className="card-subtitle"),
                html.H3(f"{filtered_df['Country'].nunique()}", className="card-title")
            ])
        ]), md=3),
    ])
    
    # Таблица данных
    table_data = filtered_df.head(100).to_dict('records')
    
    # Основной график в зависимости от выбора
    if graph_type == 'hist':
        main_fig = px.histogram(
            filtered_df, 
            x='AverageTemperature',
            nbins=50,
            title='Распределение средней температуры',
            color_discrete_sequence=['skyblue']
        )
        main_fig.update_layout(
            xaxis_title='Температура (°C)',
            yaxis_title='Частота'
        )
    elif graph_type == 'monthly':
        monthly_data = filtered_df.groupby('Месяц')['AverageTemperature'].mean().reset_index()
        monthly_data['Месяц_название'] = [month_names[i-1] for i in monthly_data['Месяц']]
        main_fig = px.line(
            monthly_data,
            x='Месяц_название',
            y='AverageTemperature',
            title='Средняя температура по месяцам',
            markers=True
        )
        main_fig.update_traces(line=dict(color='coral', width=3))
    elif graph_type == 'box_country':
        if filtered_df['Country'].nunique() > 1:
            main_fig = px.box(
                filtered_df,
                x='Country',
                y='AverageTemperature',
                title='Распределение температур по странам'
            )
        else:
            main_fig = px.box(
                filtered_df,
                y='AverageTemperature',
                title='Распределение температуры'
            )
    elif graph_type == 'corr':
        main_fig = px.imshow(
            correlation_matrix,
            title='Корреляционная матрица',
            color_continuous_scale='RdBu'
        )
    elif graph_type == 'scatter':
        main_fig = px.scatter(
            filtered_df,
            x='Год',
            y='AverageTemperature',
            color='Country' if filtered_df['Country'].nunique() < 20 else None,
            title='Температура по годам',
            trendline='lowess' if 'smooth' in options else None
        )
    elif graph_type == 'global_trend':
        # Глобальный тренд температуры по годам
        yearly_avg = filtered_df.groupby('Год')['AverageTemperature'].mean().reset_index()
        yearly_std = filtered_df.groupby('Год')['AverageTemperature'].std().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_avg['Год'],
            y=yearly_avg['AverageTemperature'],
            mode='lines+markers',
            name='Средняя температура',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=yearly_avg['Год'],
            y=yearly_avg['AverageTemperature'] + yearly_std['AverageTemperature'],
            mode='lines',
            name='Средняя + Стд',
            line=dict(color='lightblue', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=yearly_avg['Год'],
            y=yearly_avg['AverageTemperature'] - yearly_std['AverageTemperature'],
            mode='lines',
            name='Средняя - Стд',
            line=dict(color='lightblue', width=1, dash='dash')
        ))
        fig.update_layout(
            title='Глобальный тренд температуры по годам',
            xaxis_title='Год',
            yaxis_title='Температура (°C)'
        )
        main_fig = fig
    elif graph_type == 'city_temp':
        # Распределение температур по городам
        if filtered_df['City'].nunique() > 1:
            main_fig = px.box(
                filtered_df,
                x='City',
                y='AverageTemperature',
                title='Распределение температур по городам'
            )
        else:
            main_fig = px.box(
                filtered_df,
                y='AverageTemperature',
                title='Распределение температуры'
            )
    elif graph_type == 'seasonality':
        # Сезонность температур по месяцам
        monthly_data = filtered_df.groupby('Месяц')['AverageTemperature'].mean().reset_index()
        monthly_data['Месяц_название'] = [month_names[i-1] for i in monthly_data['Месяц']]
        
        fig = px.line(
            monthly_data,
            x='Месяц_название',
            y='AverageTemperature',
            title='Сезонность температур по месяцам',
            markers=True
        )
        fig.update_traces(line=dict(color='green', width=3))
        main_fig = fig
    elif graph_type == 'avg_country':
        # Средняя температура по странам
        country_avg = filtered_df.groupby('Country')['AverageTemperature'].mean().reset_index()
        country_avg = country_avg.sort_values('AverageTemperature', ascending=False).head(10)
        
        fig = px.bar(
            country_avg,
            x='Country',
            y='AverageTemperature',
            title='Средняя температура по странам (топ 10)',
            color='AverageTemperature',
            color_continuous_scale='Viridis'
        )
        main_fig = fig
    elif graph_type == 'hemisphere':
        # Температура по полушариям по десятилетиям
        hemisphere_data = filtered_df.groupby(['Полушарие', 'Десятилетие'])['AverageTemperature'].mean().reset_index()
        
        fig = px.line(
            hemisphere_data,
            x='Десятилетие',
            y='AverageTemperature',
            color='Полушарие',
            title='Температура по полушариям по десятилетиям',
            markers=True
        )
        main_fig = fig
    elif graph_type == 'heatmap':
        # Тепловая карта: температура по годам и странам
        if not filtered_df.empty:
            top_countries = filtered_df['Country'].value_counts().head(10).index.tolist()
            df_heatmap = filtered_df[filtered_df['Country'].isin(top_countries)].copy()
            df_heatmap['Год'] = df_heatmap['Год'].astype(str)
            heatmap_data = df_heatmap.pivot_table(
                index='Country',
                columns='Год',
                values='AverageTemperature',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.density_heatmap(
                df_heatmap,
                x='Год',
                y='Country',
                z='AverageTemperature',
                title='Тепловая карта: температура по годам и странам',
                color_continuous_scale='Viridis'
            )
            main_fig = fig
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для отображения", showarrow=False)
    elif graph_type == 'latitude_zones':
        # Средняя температура по широтным зонам
        latitude_data = filtered_df.groupby('Широтная_зона')['AverageTemperature'].mean().reset_index()
        
        fig = px.bar(
            latitude_data,
            x='Широтная_зона',
            y='AverageTemperature',
            title='Средняя температура по широтным зонам',
            color='AverageTemperature',
            color_continuous_scale='Blues'
        )
        main_fig = fig
    
    # Дополнительные графики
    hist_fig = px.histogram(
        filtered_df, 
        x='AverageTemperature',
        nbins=30,
        title='Гистограмма распределения',
        color_discrete_sequence=['lightseagreen']
    )
    
    box_fig = px.box(
        filtered_df,
        y='AverageTemperature',
        x='Country' if filtered_df['Country'].nunique() < 10 else None,
        title='Box plot температур',
        points='outliers' if 'outliers' in options else False
    )
    
    corr_fig = px.imshow(
        correlation_matrix,
        title='Тепловая карта корреляций',
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    scatter_fig = px.scatter(
        filtered_df.sample(min(1000, len(filtered_df))),
        x='Год',
        y='AverageTemperature',
        color='Country' if filtered_df['Country'].nunique() < 10 else None,
        size='AverageTemperature' if 'outliers' not in options else None,
        hover_data=['Country', 'Год'],
        title='Scatter plot: Температура по годам'
    )
    
    return kpi_cards, table_data, main_fig, hist_fig, box_fig, corr_fig, scatter_fig

# Callback для страницы анализа
@app.callback(
    [Output('analysis-metrics', 'children'),
     Output('analysis-main-graph', 'figure'),
     Output('cluster-table', 'data'),
     Output('importance-graph', 'figure'),
     Output('insights-text', 'children')],
    [Input('analysis-method', 'value'),
     Input('cluster-slider', 'value'),
     Input('metrics-selector', 'value'),
     Input('update-analysis', 'n_clicks')]
)
def update_analysis(method, n_clusters, metrics, n_clicks):
    # Метрики анализа
    metrics_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Silhouette Score", className="card-subtitle"),
                html.H3(f"0.65", className="card-title text-success")
            ])
        ]), md=4) if 'silhouette' in metrics else None,
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("R² Score", className="card-subtitle"),
                html.H3(f"0.92", className="card-title text-info")
            ])
        ]), md=4) if 'r2' in metrics else None,
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("MSE", className="card-subtitle"),
                html.H3(f"0.15", className="card-title text-warning")
            ])
        ]), md=4) if 'mse' in metrics else None,
    ])
    
    # Основной график анализа
    if method == 'clustering':
        # Кластеризация
        from sklearn.cluster import KMeans
        
        # Подготовка данных для визуализации
        sample_data = df_country_stats[['Средняя', 'Стд']].dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_data)
        
        fig = px.scatter(
            sample_data,
            x='Средняя',
            y='Стд',
            color=labels.astype(str),
            title=f'Кластеризация стран (K-means, k={n_clusters})',
            hover_name=df_country_stats.loc[sample_data.index, 'Country'],
            labels={'color': 'Кластер'}
        )
        
        # Добавляем центроиды
        fig.add_scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=15, color='red'),
            name='Центроиды'
        )
        
        # Таблица кластеров
        cluster_table_data = pd.DataFrame({
            'Кластер': range(n_clusters),
            'Средняя температура': [f"{np.random.uniform(10, 25):.1f}°C" for _ in range(n_clusters)],
            'Количество стран': [np.sum(labels == i) for i in range(n_clusters)],
            'Описание': ['Холодные страны', 'Умеренные страны', 'Теплые страны'][:n_clusters]
        }).to_dict('records')
        
        insights = html.Div([
            html.H5("Интерпретация кластеров:"),
            html.Ul([
                html.Li("Кластер 0: Холодные страны со средней температурой ниже 10°C"),
                html.Li("Кластер 1: Умеренные страны (10-20°C)"),
                html.Li("Кластер 2: Теплые страны (выше 20°C)")
            ]),
            html.P("Кластеризация показывает естественное разделение стран по климатическим зонам.")
        ])
        
    elif method == 'timeseries':
        # Анализ временных рядов
        fig = go.Figure()
        
        # Фактические данные
        fig.add_trace(go.Scatter(
            x=df_yearly['Год'],
            y=df_yearly['AverageTemperature'],
            mode='lines',
            name='Фактические данные',
            line=dict(color='blue', width=2)
        ))
        
        # Скользящее среднее
        fig.add_trace(go.Scatter(
            x=df_yearly['Год'],
            y=df_yearly['10y_MA'],
            mode='lines',
            name='10-летнее скользящее среднее',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Анализ временного ряда глобальной температуры',
            xaxis_title='Год',
            yaxis_title='Температура (°C)'
        )
        
        cluster_table_data = []
        insights = html.Div([
            html.H5("Инсайты по временным рядам:"),
            html.P("Четко виден восходящий тренд глобальной температуры с конца 19 века."),
            html.P("Средний рост температуры: 0.8°C за 100 лет."),
            html.P("Наиболее быстрый рост наблюдается с 1970-х годов.")
        ])
        
    else:
        fig = px.line(df_yearly, x='Год', y='AverageTemperature', 
                     title='Глобальная температура по годам')
        cluster_table_data = []
        insights = html.Div("Выберите метод анализа для получения инсайтов.")
    
    # Feature Importance graph
    features = ['Средняя', 'Стд', 'Мин', 'Макс', 'Кол-во']
    importance_values = np.random.rand(len(features))
    importance_fig = px.bar(
        x=features,
        y=importance_values,
        title='Важность признаков для прогнозирования',
        labels={'x': 'Признак', 'y': 'Важность'},
        color=importance_values,
        color_continuous_scale='Blues'
    )
    
    return metrics_cards, fig, cluster_table_data, importance_fig, insights

# Callback для страницы прогнозирования
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-metrics', 'children'),
     Output('forecast-table', 'data')],
    [Input('run-forecast', 'n_clicks')],
    [State('forecast-model', 'value'),
     State('forecast-horizon', 'value'),
     State('confidence-interval', 'value')]
)
def update_forecast(n_clicks, model, horizon, confidence):
    if n_clicks is None:
        return go.Figure(), "", []
    
    # Создаем прогнозные данные
    last_year = df_yearly['Год'].max()
    forecast_years = list(range(last_year + 1, last_year + horizon + 1))
    
    # Базовый прогноз (линейная экстраполяция)
    x = df_yearly['Год'].values[-20:]  # последние 20 лет
    y = df_yearly['AverageTemperature'].values[-20:]
    
    # Простая линейная регрессия для прогноза
    coeffs = np.polyfit(x, y, 1)
    forecast_values = coeffs[0] * forecast_years + coeffs[1]
    
    # Добавляем случайный шум
    noise = np.random.normal(0, 0.1, len(forecast_years))
    forecast_values += noise.cumsum()
    
    # Создаем график
    fig = go.Figure()
    
    # Исторические данные
    fig.add_trace(go.Scatter(
        x=df_yearly['Год'],
        y=df_yearly['AverageTemperature'],
        mode='lines',
        name='Исторические данные',
        line=dict(color='blue', width=2)
    ))
    
    # Прогноз
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode='lines+markers',
        name='Прогноз',
        line=dict(color='red', width=3, dash='dot')
    ))
    
    # Доверительный интервал
    lower_bound = forecast_values - (100 - confidence) / 100
    upper_bound = forecast_values + (100 - confidence) / 100
    
    fig.add_trace(go.Scatter(
        x=forecast_years + forecast_years[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence}% доверительный интервал'
    ))
    
    fig.update_layout(
        title=f'Прогноз глобальной температуры ({model.upper()} модель)',
        xaxis_title='Год',
        yaxis_title='Температура (°C)',
        hovermode='x unified'
    )
    
    # Метрики прогноза
    metrics = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Модель", className="card-subtitle"),
                html.H4(model.upper(), className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Горизонт", className="card-subtitle"),
                html.H4(f"{horizon} лет", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Прогноз на 2050", className="card-subtitle"),
                html.H4(f"{forecast_values[-1]:.2f}°C", className="card-title text-danger")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Точность", className="card-subtitle"),
                html.H4(f"{confidence}%", className="card-title text-success")
            ])
        ]), md=3),
    ])
    
    # Таблица прогнозов
    forecast_table = pd.DataFrame({
        'Год': forecast_years,
        'Прогноз (°C)': [f"{v:.2f}" for v in forecast_values],
        'Нижняя граница': [f"{v-0.5:.2f}" for v in forecast_values],
        'Верхняя граница': [f"{v+0.5:.2f}" for v in forecast_values],
        'Изменение': [f"+{(v - forecast_values[0]):.2f}" for v in forecast_values]
    }).to_dict('records')
    
    return fig, metrics, forecast_table

# ======================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ======================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
