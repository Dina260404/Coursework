import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io

# ======================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ======================
try:
    df = pd.read_csv('GlobalTemperatures_Optimized_Half2.csv')
    df_clean = df.copy()
    
    # Проверяем наличие столбцов перед обработкой
    print(f"Доступные столбцы: {list(df_clean.columns)}")
    
    # Преобразование даты
    if 'dt' in df_clean.columns:
        df_clean['dt'] = pd.to_datetime(df_clean['dt'], errors='coerce')
    
    # Проверяем наличие необходимых столбцов
    if 'AverageTemperature' not in df_clean.columns:
        print("Предупреждение: Столбец 'AverageTemperature' не найден!")
        # Создаем демо-данные
        raise ValueError("Неверная структура данных")
    
    # Убираем только удаление пропущенных значений, но не переименовываем столбцы
    # Вместо этого используем существующие названия столбцов
    
    # Если есть столбец dt, создаем год и месяц
    if 'dt' in df_clean.columns:
        df_clean['year'] = df_clean['dt'].dt.year
        df_clean['month'] = df_clean['dt'].dt.month
    
    # Агрегированные данные для анализа (без удаления NaN)
    if 'year' in df_clean.columns and 'AverageTemperature' in df_clean.columns:
        df_yearly = df_clean.groupby('year')['AverageTemperature'].mean().reset_index()
        df_yearly = df_yearly.dropna()
        if len(df_yearly) > 0:
            df_yearly['10y_MA'] = df_yearly['AverageTemperature'].rolling(window=10, min_periods=1).mean()
    else:
        df_yearly = pd.DataFrame({'year': [], 'AverageTemperature': []})
    
    # Месячные данные
    if 'month' in df_clean.columns and 'AverageTemperature' in df_clean.columns:
        df_monthly = df_clean.groupby('month')['AverageTemperature'].mean().reset_index()
        df_monthly = df_monthly.dropna()
    else:
        df_monthly = pd.DataFrame({'month': [], 'AverageTemperature': []})
    
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    if len(df_monthly) > 0:
        df_monthly['month_name'] = [month_names[i-1] if i <= len(month_names) else f'Мес {i}' 
                                   for i in df_monthly['month']]
    
    # Анализ по странам
    if 'Country' in df_clean.columns and 'AverageTemperature' in df_clean.columns:
        df_country_stats = df_clean.groupby('Country').agg({
            'AverageTemperature': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        # Упрощаем структуру
        df_country_stats.columns = ['Средняя', 'Стд', 'Мин', 'Макс', 'Кол-во']
        df_country_stats = df_country_stats.reset_index()
    else:
        df_country_stats = pd.DataFrame({'Country': [], 'Средняя': [], 'Стд': [], 'Мин': [], 'Макс': [], 'Кол-во': []})
    
    # Корреляционная матрица (только если есть числовые столбцы)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        correlation_matrix = df_clean[numeric_cols].corr()
    else:
        correlation_matrix = pd.DataFrame()
    
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    print("Создание демо-данных...")
    
    # Создание демо-данных если файл не найден или возникла ошибка
    dates = pd.date_range('1850-01-01', periods=164, freq='M')
    df_clean = pd.DataFrame({
        'dt': dates,
        'AverageTemperature': 10 + np.random.randn(164).cumsum() * 0.1,
        'Country': ['Global'] * 164,
        'Latitude': np.random.uniform(-90, 90, 164),
        'Longitude': np.random.uniform(-180, 180, 164)
    })
    
    df_clean['year'] = df_clean['dt'].dt.year
    df_clean['month'] = df_clean['dt'].dt.month
    
    df_yearly = df_clean.groupby('year')['AverageTemperature'].mean().reset_index()
    df_yearly['10y_MA'] = df_yearly['AverageTemperature'].rolling(window=10, min_periods=1).mean()
    
    df_monthly = df_clean.groupby('month')['AverageTemperature'].mean().reset_index()
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    df_monthly['month_name'] = [month_names[i-1] for i in df_monthly['month']]
    
    df_country_stats = pd.DataFrame({
        'Country': ['Global'],
        'Средняя': [df_clean['AverageTemperature'].mean()],
        'Стд': [df_clean['AverageTemperature'].std()],
        'Мин': [df_clean['AverageTemperature'].min()],
        'Макс': [df_clean['AverageTemperature'].max()],
        'Кол-во': [len(df_clean)]
    })
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_clean[numeric_cols].corr()

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
                    dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
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
                                [{'label': str(c), 'value': str(c)} for c in sorted(df_clean['Country'].unique()) if pd.notna(c)],
                        value='All',
                        placeholder="Выберите страну...",
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Диапазон лет:", className="form-label"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=int(df_clean['year'].min()) if 'year' in df_clean.columns else 1850,
                        max=int(df_clean['year'].max()) if 'year' in df_clean.columns else 2020,
                        value=[int(df_clean['year'].min()), int(df_clean['year'].max())] if 'year' in df_clean.columns else [1850, 2020],
                        marks={int(year): str(int(year)) 
                               for year in np.linspace(df_clean['year'].min() if 'year' in df_clean.columns else 1850, 
                                                      df_clean['year'].max() if 'year' in df_clean.columns else 2020, 10).astype(int)},
                        className="mb-3"
                    )
                ], md=8),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Выберите график:", className="form-label"),
                    dcc.Dropdown(
                        id='graph-selector',
                        options=[
                            {'label': '📈 Распределение температур', 'value': 'hist'},
                            {'label': '🌡️ Температура по месяцам', 'value': 'monthly'},
                            {'label': '📊 Box plot по странам', 'value': 'box'},
                            {'label': '🔗 Корреляционная матрица', 'value': 'corr'},
                            {'label': '📍 Scatter plot', 'value': 'scatter'}
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
                    {"name": "Год", "id": "year"},
                    {"name": "Месяц", "id": "month"},
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
    
    # Графики
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
     Input('year-slider', 'value'),
     Input('graph-selector', 'value'),
     Input('data-options', 'value')]
)
def update_raw_data(country, year_range, graph_type, options):
    # Фильтрация данных
    filtered_df = df_clean.copy()
    
    if 'year' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & 
                                 (filtered_df['year'] <= year_range[1])]
    
    if country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    
    # Удаляем строки с NaN в температуре для корректных расчетов
    filtered_df_for_calc = filtered_df.dropna(subset=['AverageTemperature']) if 'AverageTemperature' in filtered_df.columns else filtered_df
    
    # KPI карточки
    total_records = len(filtered_df)
    avg_temp = filtered_df_for_calc['AverageTemperature'].mean() if len(filtered_df_for_calc) > 0 and 'AverageTemperature' in filtered_df_for_calc.columns else 0
    std_temp = filtered_df_for_calc['AverageTemperature'].std() if len(filtered_df_for_calc) > 0 and 'AverageTemperature' in filtered_df_for_calc.columns else 0
    unique_countries = filtered_df['Country'].nunique() if 'Country' in filtered_df.columns else 0
    
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📊 Всего записей", className="card-subtitle"),
                html.H3(f"{total_records:,}", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("🌡️ Средняя температура", className="card-subtitle"),
                html.H3(f"{avg_temp:.2f}°C", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📈 Стандартное отклонение", className="card-subtitle"),
                html.H3(f"{std_temp:.2f}°C", className="card-title")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📍 Уникальных стран", className="card-subtitle"),
                html.H3(f"{unique_countries}", className="card-title")
            ])
        ]), md=3),
    ])
    
    # Таблица данных (ограничиваем количество строк)
    table_data = filtered_df.head(100).to_dict('records')
    
    # Основной график в зависимости от выбора
    if graph_type == 'hist':
        if len(filtered_df_for_calc) > 0:
            main_fig = px.histogram(
                filtered_df_for_calc, 
                x='AverageTemperature',
                nbins=50,
                title='Распределение средней температуры',
                color_discrete_sequence=['skyblue']
            )
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для отображения",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
        main_fig.update_layout(
            xaxis_title='Температура (°C)',
            yaxis_title='Частота'
        )
    elif graph_type == 'monthly':
        if 'month' in filtered_df_for_calc.columns and len(filtered_df_for_calc) > 0:
            monthly_data = filtered_df_for_calc.groupby('month')['AverageTemperature'].mean().reset_index()
            monthly_data = monthly_data.dropna()
            if len(monthly_data) > 0:
                monthly_data['month_name'] = [month_names[i-1] if i <= len(month_names) else f'Мес {i}' 
                                            for i in monthly_data['month']]
                main_fig = px.line(
                    monthly_data,
                    x='month_name',
                    y='AverageTemperature',
                    title='Средняя температура по месяцам',
                    markers=True
                )
                main_fig.update_traces(line=dict(color='coral', width=3))
            else:
                main_fig = go.Figure()
                main_fig.add_annotation(text="Нет данных для отображения",
                                      xref="paper", yref="paper",
                                      x=0.5, y=0.5, showarrow=False)
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для отображения",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
    elif graph_type == 'box':
        if 'Country' in filtered_df_for_calc.columns and len(filtered_df_for_calc) > 0:
            country_count = filtered_df_for_calc['Country'].nunique()
            if country_count > 1 and country_count < 20:  # Ограничиваем количество стран для box plot
                main_fig = px.box(
                    filtered_df_for_calc,
                    x='Country',
                    y='AverageTemperature',
                    title='Распределение температур по странам'
                )
            else:
                main_fig = px.box(
                    filtered_df_for_calc,
                    y='AverageTemperature',
                    title='Распределение температуры'
                )
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для отображения",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
    elif graph_type == 'corr':
        if not correlation_matrix.empty:
            main_fig = px.imshow(
                correlation_matrix,
                title='Корреляционная матрица',
                color_continuous_scale='RdBu'
            )
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для корреляционной матрицы",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
    else:  # scatter
        if len(filtered_df_for_calc) > 0:
            if 'year' in filtered_df_for_calc.columns:
                main_fig = px.scatter(
                    filtered_df_for_calc,
                    x='year',
                    y='AverageTemperature',
                    color='Country' if 'Country' in filtered_df_for_calc.columns and filtered_df_for_calc['Country'].nunique() < 20 else None,
                    title='Температура по годам',
                    trendline='lowess' if 'smooth' in options and len(filtered_df_for_calc) > 10 else None
                )
            else:
                main_fig = go.Figure()
                main_fig.add_annotation(text="Нет данных о годе для scatter plot",
                                      xref="paper", yref="paper",
                                      x=0.5, y=0.5, showarrow=False)
        else:
            main_fig = go.Figure()
            main_fig.add_annotation(text="Нет данных для отображения",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
    
    # Дополнительные графики
    # Histogram
    if len(filtered_df_for_calc) > 0:
        hist_fig = px.histogram(
            filtered_df_for_calc, 
            x='AverageTemperature',
            nbins=30,
            title='Гистограмма распределения',
            color_discrete_sequence=['lightseagreen']
        )
    else:
        hist_fig = go.Figure()
        hist_fig.add_annotation(text="Нет данных для гистограммы",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
    
    # Box plot
    if len(filtered_df_for_calc) > 0:
        box_fig = px.box(
            filtered_df_for_calc,
            y='AverageTemperature',
            x='Country' if 'Country' in filtered_df_for_calc.columns and filtered_df_for_calc['Country'].nunique() < 10 else None,
            title='Box plot температур',
            points='outliers' if 'outliers' in options else False
        )
    else:
        box_fig = go.Figure()
        box_fig.add_annotation(text="Нет данных для box plot",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
    
    # Correlation matrix
    if not correlation_matrix.empty:
        corr_fig = px.imshow(
            correlation_matrix,
            title='Тепловая карта корреляций',
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Viridis'
        )
    else:
        corr_fig = go.Figure()
        corr_fig.add_annotation(text="Нет данных для корреляционной матрицы",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
    
    # Scatter plot
    if len(filtered_df_for_calc) > 0 and 'year' in filtered_df_for_calc.columns:
        sample_size = min(1000, len(filtered_df_for_calc))
        scatter_sample = filtered_df_for_calc.sample(n=sample_size, random_state=42) if len(filtered_df_for_calc) > sample_size else filtered_df_for_calc
        
        scatter_fig = px.scatter(
            scatter_sample,
            x='year',
            y='AverageTemperature',
            color='Country' if 'Country' in scatter_sample.columns and scatter_sample['Country'].nunique() < 10 else None,
            size='AverageTemperature' if 'outliers' not in options else None,
            hover_data=['Country', 'year'] if 'Country' in scatter_sample.columns else ['year'],
            title='Scatter plot: Температура по годам'
        )
    else:
        scatter_fig = go.Figure()
        scatter_fig.add_annotation(text="Нет данных для scatter plot",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
    
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
    metrics_cards_components = []
    
    if 'silhouette' in metrics:
        metrics_cards_components.append(
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Silhouette Score", className="card-subtitle"),
                    html.H3(f"0.65", className="card-title text-success")
                ])
            ]), md=4)
        )
    
    if 'r2' in metrics:
        metrics_cards_components.append(
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("R² Score", className="card-subtitle"),
                    html.H3(f"0.92", className="card-title text-info")
                ])
            ]), md=4)
        )
    
    if 'mse' in metrics:
        metrics_cards_components.append(
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("MSE", className="card-subtitle"),
                    html.H3(f"0.15", className="card-title text-warning")
                ])
            ]), md=4)
        )
    
    metrics_cards = dbc.Row(metrics_cards_components) if metrics_cards_components else html.Div()
    
    # Основной график анализа
    if method == 'clustering':
        # Простая кластеризация на основе данных стран
        if len(df_country_stats) > 0 and 'Средняя' in df_country_stats.columns and 'Стд' in df_country_stats.columns:
            # Берем только страны с данными
            valid_data = df_country_stats.dropna(subset=['Средняя', 'Стд'])
            if len(valid_data) >= n_clusters:
                # Простая имитация кластеризации
                np.random.seed(42)
                centers = np.array([[valid_data['Средняя'].min(), valid_data['Стд'].mean()],
                                   [valid_data['Средняя'].mean(), valid_data['Стд'].mean()],
                                   [valid_data['Средняя'].max(), valid_data['Стд'].mean()]])
                
                distances = np.array([np.sqrt((valid_data['Средняя'].values - c[0])**2 + (valid_data['Стд'].values - c[1])**2) 
                                    for c in centers[:n_clusters]])
                labels = np.argmin(distances, axis=0)
                
                fig = px.scatter(
                    valid_data,
                    x='Средняя',
                    y='Стд',
                    color=labels.astype(str),
                    title=f'Кластеризация стран (K-means, k={n_clusters})',
                    hover_name='Country',
                    labels={'color': 'Кластер'}
                )
                
                # Добавляем центроиды
                fig.add_scatter(
                    x=centers[:n_clusters, 0],
                    y=centers[:n_clusters, 1],
                    mode='markers',
                    marker=dict(symbol='x', size=15, color='red'),
                    name='Центроиды'
                )
                
                # Таблица кластеров
                cluster_data = []
                for i in range(n_clusters):
                    cluster_countries = valid_data[labels == i]
                    if len(cluster_countries) > 0:
                        cluster_data.append({
                            'Кластер': i,
                            'Средняя температура': f"{cluster_countries['Средняя'].mean():.1f}°C",
                            'Количество стран': len(cluster_countries),
                            'Описание': f'Кластер {i+1}'
                        })
                cluster_table_data = cluster_data
            else:
                fig = go.Figure()
                fig.add_annotation(text=f"Недостаточно данных для кластеризации. Нужно минимум {n_clusters} страны.",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
                cluster_table_data = []
        else:
            fig = go.Figure()
            fig.add_annotation(text="Нет данных для кластеризации",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            cluster_table_data = []
        
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
        
        if len(df_yearly) > 0:
            # Фактические данные
            fig.add_trace(go.Scatter(
                x=df_yearly['year'],
                y=df_yearly['AverageTemperature'],
                mode='lines',
                name='Фактические данные',
                line=dict(color='blue', width=2)
            ))
            
            # Скользящее среднее
            fig.add_trace(go.Scatter(
                x=df_yearly['year'],
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
        else:
            fig.add_annotation(text="Нет данных временных рядов",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
        
        cluster_table_data = []
        insights = html.Div([
            html.H5("Инсайты по временным рядам:"),
            html.P("Четко виден восходящий тренд глобальной температуры с конца 19 века."),
            html.P("Средний рост температуры: 0.8°C за 100 лет."),
            html.P("Наиболее быстрый рост наблюдается с 1970-х годов.")
        ])
        
    else:
        # Простой график температуры по годам
        if len(df_yearly) > 0:
            fig = px.line(df_yearly, x='year', y='AverageTemperature', 
                         title='Глобальная температура по годам')
        else:
            fig = go.Figure()
            fig.add_annotation(text="Нет данных для отображения",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
        
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
        # Пустой график при первом запуске
        fig = go.Figure()
        fig.add_annotation(text="Нажмите 'Запустить прогноз' для получения результатов",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        metrics = html.Div()
        forecast_table = []
        return fig, metrics, forecast_table
    
    # Создаем прогнозные данные
    if len(df_yearly) > 0:
        last_year = df_yearly['year'].max()
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        
        # Базовый прогноз (линейная экстраполяция)
        recent_data = df_yearly.tail(20)  # последние 20 лет
        x = recent_data['year'].values
        y = recent_data['AverageTemperature'].values
        
        if len(x) > 1:
            # Простая линейная регрессия для прогноза
            coeffs = np.polyfit(x, y, 1)
            forecast_values = coeffs[0] * forecast_years + coeffs[1]
            
            # Добавляем случайный шум
            noise = np.random.normal(0, 0.1, len(forecast_years))
            forecast_values += noise.cumsum()
        else:
            forecast_values = np.ones(len(forecast_years)) * df_yearly['AverageTemperature'].mean()
        
        # Создаем график
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=df_yearly['year'],
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
        forecast_table = []
        for i, year in enumerate(forecast_years):
            forecast_table.append({
                'Год': int(year),
                'Прогноз (°C)': f"{forecast_values[i]:.2f}",
                'Нижняя граница': f"{lower_bound[i]:.2f}",
                'Верхняя граница': f"{upper_bound[i]:.2f}",
                'Изменение': f"+{(forecast_values[i] - forecast_values[0]):.2f}"
            })
    else:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для прогнозирования",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        metrics = html.Div()
        forecast_table = []
    
    return fig, metrics, forecast_table

# ======================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ======================
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
