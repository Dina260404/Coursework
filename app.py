import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ======================
# ЗАГРУЗКА ДАННЫХ
# ======================
df = pd.read_csv('GlobalTemperatures_Optimized_Half2.csv')

# Приведение числовых колонок к числу (только те, что есть)
df['Год'] = pd.to_numeric(df['Год'], errors='coerce')
df['СредняяТемпература'] = pd.to_numeric(df['СредняяТемпература'], errors='coerce')

# Разделение по типам
df_global = df[df['Тип'] == 'global_yearly'].copy()
df_countries = df[df['Тип'] == 'country'].copy()
df_monthly = df[df['Тип'] == 'global_monthly'].copy()
df_hemi = df[df['Тип'] == 'hemisphere_yearly'].copy()

# Уникальные страны (если есть)
countries = ['All']
if not df_countries.empty:
    countries += sorted(df_countries['Страна'].dropna().unique().tolist())

# Годы для слайдера
all_years = []
if not df_global.empty:
    all_years.extend(df_global['Год'].dropna().astype(int).tolist())
if not df_hemi.empty:
    all_years.extend(df_hemi['Год'].dropna().astype(int).tolist())
years = sorted(set(all_years)) if all_years else [1850, 2013]

# ======================
# ИНИЦИАЛИЗАЦИЯ DASH
# ======================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"]
)
server = app.server  # ← ОБЯЗАТЕЛЬНО для Render

# ======================
# LAYOUT
# ======================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("🌍 Environmental Impact Monitor", className="text-center my-4"),
        html.Div([
            dcc.Link("📊 Raw Data Visualization", href="/", className="btn btn-outline-primary m-2"),
            dcc.Link("🔍 Analysis Results", href="/analysis", className="btn btn-outline-success m-2")
        ], className="text-center mb-4")
    ]),
    html.Div(id='page-content')
])

# Страница 1: Визуализация данных
raw_layout = html.Div([
    html.H2("📊 Raw Data Visualization", className="text-center mb-4"),
    
    # Фильтры
    html.Div([
        html.Div([
            html.Label("Страна:", className="form-label"),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': c, 'value': c} for c in countries],
                value='All',
                className="form-control"
            )
        ], className="col-md-4"),
        html.Div([
            html.Label("Годы:", className="form-label"),
            dcc.RangeSlider(
                id='year-slider',
                min=min(years),
                max=max(years),
                value=[min(years), max(years)],
                marks={y: str(y) for y in range(min(years), max(years)+1, 20)},
                className="mt-2"
            )
        ], className="col-md-8")
    ], className="row mb-4"),

    # KPI-карточки
    html.Div(id='kpi-cards', className="row mb-4"),

    # Таблица
    html.Div([
        dash_table.DataTable(
            id='data-table',
            columns=[
                {"name": "Тип", "id": "Тип"},
                {"name": "Год", "id": "Год"},
                {"name": "Страна", "id": "Страна"},
                {"name": "Полушарие", "id": "Полушарие"},
                {"name": "Средняя температура", "id": "СредняяТемпература"}
            ],
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'}
        )
    ], className="mb-4"),

    # Графики
    html.Div([
        html.Div(dcc.Graph(id='hist-plot'), className="col-md-6"),
        html.Div(dcc.Graph(id='box-plot'), className="col-md-6"),
    ], className="row mb-4"),

    html.Div(dcc.Graph(id='scatter-plot'), className="mb-4"),
])

# Страница 2: Анализ
analysis_layout = html.Div([
    html.H2("🔍 Analysis Results", className="text-center mb-4"),
    html.Div([
        html.Div([
            html.Label("Выбор модели:", className="form-label"),
            dcc.RadioItems(
                id='model-selector',
                options=[
                    {'label': 'Глобальный тренд', 'value': 'trend'},
                    {'label': 'Сравнение полушарий', 'value': 'hemisphere'}
                ],
                value='trend',
                labelStyle={'display': 'block'}
            )
        ], className="col-md-3"),
        html.Div(id='metrics-cards', className="col-md-9")
    ], className="row mb-4"),
    html.Div(dcc.Graph(id='analysis-graph'), className="mb-4"),
    html.Div(id='insights-text', className="alert alert-info")
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
    Output('data-table', 'data'),
    Output('kpi-cards', 'children'),
    Output('hist-plot', 'figure'),
    Output('box-plot', 'figure'),
    Output('scatter-plot', 'figure'),
    Input('country-filter', 'value'),
    Input('year-slider', 'value')
)
def update_raw_data(country, year_range):
    # Объединяем данные
    dff = pd.concat([df_global, df_countries, df_hemi], ignore_index=True)
    dff = dff[(dff['Год'] >= year_range[0]) & (dff['Год'] <= year_range[1])]
    if country != 'All':
        dff = dff[dff['Страна'] == country]
    dff = dff.dropna(subset=['СредняяТемпература'])

    # KPI
    kpi_cards = [
        html.Div(html.Div([
            html.H5("Записей", className="card-title"),
            html.H4(f"{len(dff):,}", className="card-text")
        ], className="card-body"), className="col-md-3")
    ]
    if len(dff) > 0:
        kpi_cards.append(
            html.Div(html.Div([
                html.H5("Ср. температура", className="card-title"),
                html.H4(f"{dff['СредняяТемпература'].mean():.2f}°C", className="card-text")
            ], className="card-body"), className="col-md-3")
        )

    # Таблица
    table_cols = ['Тип', 'Год', 'Страна', 'Полушарие', 'СредняяТемпература']
    table_data = dff[table_cols].dropna(how='all').fillna('').head(50).to_dict('records')

    # Гистограмма
    hist = px.histogram(dff, x='СредняяТемпература', nbins=20, title="Распределение температур")

    # Box-plot
    box = px.box(dff, y='СредняяТемпература', title="Разброс температур")

    # Scatter
    scatter = px.scatter(
        dff, x='Год', y='СредняяТемпература',
        color='Тип', hover_data=['Страна', 'Полушарие'],
        title="Температура по годам"
    )

    return table_data, kpi_cards, hist, box, scatter

@app.callback(
    Output('analysis-graph', 'figure'),
    Output('metrics-cards', 'children'),
    Output('insights-text', 'children'),
    Input('model-selector', 'value')
)
def update_analysis(model):
    if model == 'hemisphere' and not df_hemi.empty:
        fig = px.line(
            df_hemi,
            x='Год',
            y='СредняяТемпература',
            color='Полушарие',
            title="Сравнение температур: Северное vs Южное полушарие"
        )
        insights = "Северное полушарие нагревается быстрее из-за большей концентрации суши и промышленности."
        metrics = []
    else:
        fig = px.line(
            df_global,
            x='Год',
            y='СредняяТемпература',
            title="Глобальный тренд средней температуры (1850–2013)"
        )
        if len(df_global) > 5:
            fig.add_scatter(
                x=df_global['Год'],
                y=df_global['СредняяТемпература'].rolling(window=10, min_periods=1).mean(),
                mode='lines',
                name='10-летнее скользящее среднее'
            )
        insights = "Средняя глобальная температура выросла более чем на 1°C с середины XIX века."
        metrics = []

    return fig, metrics, insights

# ======================
# ЗАПУСК
# ======================
if __name__ == '__main__':
    app.run_server(debug=True)
