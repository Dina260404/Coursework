import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd

# ---------- Загрузка и обработка данных ----------
df = pd.read_csv("data/GlobalTemperatures_Optimized_Half2_fixed.csv", parse_dates=["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Обработка координат
df["Latitude"] = df["Latitude"].str.replace("N", "").str.replace("S", "-").astype(float)
df["Longitude"] = df["Longitude"].str.replace("E", "").str.replace("W", "-").astype(float)

# Уникальные страны и города для фильтров
countries = sorted(df["Country"].dropna().unique())
cities = sorted(df["City"].dropna().unique())
min_year = df["Year"].min()
max_year = df["Year"].max()

# ---------- Инициализация приложения ----------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

server = app.server  # для Render

# ---------- Макет навбара ----------
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Raw Data", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Analysis Results", href="/analysis", active="exact")),
    ],
    brand="🌍 Environmental Impact Monitor",
    brand_href="/",
    color="primary",
    dark=True,
    fixed="top",
)

# ---------- Контент страниц ----------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', style={"marginTop": "60px", "padding": "20px"})
])

# ---------- Страница 1: Raw Data Visualization ----------
page_raw = html.Div([
    html.H2("🌡️ Raw Data Visualization", className="mb-4"),

    # Фильтры
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': c, 'value': c} for c in countries],
                placeholder="Select Country",
                multi=True
            )
        ], md=3),
        dbc.Col([
            dcc.Dropdown(
                id='city-filter',
                options=[{'label': c, 'value': c} for c in cities],
                placeholder="Select City",
                multi=True
            )
        ], md=3),
        dbc.Col([
            dcc.RangeSlider(
                id='year-slider',
                min=min_year,
                max=max_year,
                step=1,
                value=[min_year, max_year],
                marks={str(year): str(year) for year in range(min_year, max_year+1, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=6),
    ], className="mb-4"),

    # KPI карточки
    html.Div(id='kpi-cards', className="mb-4"),

    # Таблица
    html.Div([
        dash.dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': 12},
        )
    ], className="mb-4"),

    # Визуализации
    dbc.Row([
        dbc.Col(dcc.Graph(id='temp-hist'), md=6),
        dbc.Col(dcc.Graph(id='temp-box'), md=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='country-bar'), md=6),
        dbc.Col(dcc.Graph(id='corr-heatmap'), md=6),
    ], className="mb-4"),

    # Карта
    dcc.Graph(id='map-plot')
])

# ---------- Страница 2: Analysis Results ----------
page_analysis = html.Div([
    html.H2("🔍 Analysis Results & Insights", className="mb-4"),

    # Фильтры (повторяем те же)
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='analysis-country-filter',
                options=[{'label': c, 'value': c} for c in countries],
                placeholder="Select Country",
                multi=True
            )
        ], md=3),
        dbc.Col([
            dcc.Dropdown(
                id='analysis-city-filter',
                options=[{'label': c, 'value': c} for c in cities],
                placeholder="Select City",
                multi=True
            )
        ], md=3),
        dbc.Col([
            dcc.RangeSlider(
                id='analysis-year-slider',
                min=min_year,
                max=max_year,
                step=1,
                value=[min_year, max_year],
                marks={str(year): str(year) for year in range(min_year, max_year+1, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=6),
    ], className="mb-4"),

    # KPI Analysis
    html.Div(id='analysis-kpi-cards', className="mb-4"),

    # Тренд
    dcc.Graph(id='temp-trend'),

    # Кластеризация (упрощённая: по широте и температуре)
    dcc.Graph(id='cluster-map'),

    # Динамический insight
    html.Div(id='dynamic-insight', className="mt-3 p-3 bg-light rounded")
])

# ---------- Коллбэки для фильтрации ----------
def filter_data(country, city, year_range):
    dff = df.copy()
    if country:
        dff = dff[dff["Country"].isin(country)]
    if city:
        dff = dff[dff["City"].isin(city)]
    dff = dff[(dff["Year"] >= year_range[0]) & (dff["Year"] <= year_range[1])]
    return dff

# ---------- Raw Data Callbacks ----------
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('data-table', 'data'),
     Output('temp-hist', 'figure'),
     Output('temp-box', 'figure'),
     Output('country-bar', 'figure'),
     Output('corr-heatmap', 'figure'),
     Output('map-plot', 'figure')],
    [Input('country-filter', 'value'),
     Input('city-filter', 'value'),
     Input('year-slider', 'value')]
)
def update_raw(country, city, year_range):
    dff = filter_data(country, city, year_range)

    # KPI
    total = len(dff)
    missing = dff.isnull().sum().sum()
    mean_temp = dff["AverageTemperature"].mean()
    std_temp = dff["AverageTemperature"].std()

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Records"), html.H4(f"{total:,}")])], color="light")),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Missing Values"), html.H4(f"{missing:,}")])], color="warning")),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Mean Temp (°C)"), html.H4(f"{mean_temp:.2f}")])], color="info")),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Std Dev"), html.H4(f"{std_temp:.2f}")])], color="secondary")),
    ])

    # Визуализации
    import plotly.express as px
    temp_hist = px.histogram(dff, x="AverageTemperature", nbins=30, title="🌡️ Temperature Distribution")
    temp_box = px.box(dff, y="AverageTemperature", color="Country", title="🌡️ Temp by Country")
    country_bar = px.histogram(dff, y="Country", title="📍 Records per Country")
    
    corr_cols = ["AverageTemperature", "Year", "Latitude", "Longitude"]
    corr_data = dff[corr_cols].corr()
    corr_heatmap = px.imshow(corr_data, text_auto=True, title="🔗 Correlation Heatmap")

    map_fig = px.scatter_mapbox(
        dff,
        lat="Latitude",
        lon="Longitude",
        color="AverageTemperature",
        size="AverageTemperatureUncertainty",
        hover_name="City",
        hover_data=["Country", "Date", "AverageTemperature"],
        zoom=1,
        title="🌍 Global Temperature Observations",
        mapbox_style="open-street-map"
    )
    map_fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

    return (
        kpi_cards,
        dff.to_dict('records'),
        temp_hist,
        temp_box,
        country_bar,
        corr_heatmap,
        map_fig
    )

# ---------- Analysis Callbacks ----------
@app.callback(
    [Output('analysis-kpi-cards', 'children'),
     Output('temp-trend', 'figure'),
     Output('cluster-map', 'figure'),
     Output('dynamic-insight', 'children')],
    [Input('analysis-country-filter', 'value'),
     Input('analysis-city-filter', 'value'),
     Input('analysis-year-slider', 'value')]
)
def update_analysis(country, city, year_range):
    dff = filter_data(country, city, year_range)

    # Тренд: средняя температура по годам
    yearly = dff.groupby("Year")["AverageTemperature"].mean().reset_index()
    import plotly.express as px
    trend_fig = px.line(yearly, x="Year", y="AverageTemperature", title="📈 Global Avg Temperature Trend")
    trend_fig.add_scatter(x=yearly["Year"], y=yearly["AverageTemperature"].rolling(window=5).mean(),
                          mode='lines', name='5-Year Rolling Avg', line=dict(dash='dash'))

    # Упрощённая кластеризация: 3 группы по температуре
    from sklearn.cluster import KMeans
    if len(dff) < 3:
        cluster_fig = px.scatter_mapbox(title="⚠️ Not enough data for clustering")
        cluster_fig.update_layout(mapbox_style="open-street-map")
        insight = "Insufficient data to compute clusters."
    else:
        kmeans = KMeans(n_clusters=3, n_init=10).fit(dff[["AverageTemperature", "Latitude"]])
        dff["Cluster"] = kmeans.labels_
        cluster_fig = px.scatter_mapbox(
            dff,
            lat="Latitude",
            lon="Longitude",
            color="Cluster",
            hover_name="City",
            hover_data=["AverageTemperature"],
            title="📍 Temperature Clusters (K=3)",
            mapbox_style="open-street-map"
        )
        # Инсайт
        cluster_means = dff.groupby("Cluster")["AverageTemperature"].mean()
        max_cluster = cluster_means.idxmax()
        min_cluster = cluster_means.idxmin()
        diff = cluster_means[max_cluster] - cluster_means[min_cluster]
        insight = f"In this selection, Cluster {max_cluster} is on average {diff:.1f}°C warmer than Cluster {min_cluster}."

    # Метрики
    r2 = None
    if len(yearly) > 1:
        from sklearn.linear_model import LinearRegression
        X = yearly[["Year"]]
        y = yearly["AverageTemperature"]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

    analysis_kpi = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Trend R²"), html.H4(f"{r2:.3f}" if r2 else "–")])], color="success")),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Clusters"), html.H4("3")])], color="primary")),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Cities"), html.H4(f"{dff['City'].nunique()}")])], color="info")),
    ])

    return analysis_kpi, trend_fig, cluster_fig, insight

# ---------- Навигация ----------
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/analysis":
        return page_analysis
    return page_raw

# ---------- Запуск ----------
if __name__ == '__main__':
    app.run_server(debug=True)
