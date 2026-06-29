from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from flask import Flask

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_disease_clean.csv"

server = Flask(__name__)
app = Dash(__name__, server=server, title="Analisis de enfermedad cardiaca")


LABELS = {
    "age": "Edad",
    "sex": "Sexo",
    "cp": "Tipo de dolor de pecho",
    "trestbps": "Presion arterial en reposo",
    "chol": "Colesterol",
    "fbs": "Glucosa en ayunas > 120",
    "restecg": "ECG en reposo",
    "thalach": "Frecuencia cardiaca maxima",
    "exang": "Angina inducida por ejercicio",
    "oldpeak": "Depresion ST",
    "slope": "Pendiente ST",
    "ca": "Vasos coloreados",
    "thal": "Thal",
    "num": "Severidad original",
}

CATEGORY_LABELS = {
    "sex": {0: "Mujer", 1: "Hombre"},
    "cp": {
        1: "Angina tipica",
        2: "Angina atipica",
        3: "Dolor no anginoso",
        4: "Asintomatico",
    },
    "fbs": {0: "No", 1: "Si"},
    "restecg": {0: "Normal", 1: "Anomalia ST-T", 2: "Hipertrofia probable"},
    "exang": {0: "No", 1: "Si"},
    "slope": {1: "Ascendente", 2: "Plana", 3: "Descendente"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"},
}

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["disease"] = np.where(df["num"] > 0, "Con enfermedad", "Sin enfermedad")
    df["severity"] = df["num"].map(
        {
            0: "0 - Sin enfermedad",
            1: "1 - Leve",
            2: "2 - Moderada",
            3: "3 - Alta",
            4: "4 - Muy alta",
        }
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 39, 49, 59, 69, 120],
        labels=["<40", "40-49", "50-59", "60-69", "70+"],
        include_lowest=True,
    )
    for column, mapping in CATEGORY_LABELS.items():
        df[f"{column}_label"] = df[column].map(mapping).fillna(df[column].astype(str))
    return df


df = load_data()


def target_column(mode: str) -> str:
    return "severity" if mode == "severity" else "disease"


def label_for(column: str) -> str:
    return LABELS.get(column, column)


def metric_card(title: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Span(title, className="metric-title"),
            html.Strong(value, className="metric-value"),
            html.Span(note, className="metric-note"),
        ],
        className="metric-card",
    )


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        xref="paper",
        yref="paper",
        font={"size": 16},
    )
    fig.update_layout(template="plotly_white", xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def make_summary_cards(filtered: pd.DataFrame) -> list[html.Div]:
    disease_rate = (filtered["num"].gt(0).mean() * 100) if len(filtered) else 0
    return [
        metric_card("Registros", f"{len(filtered):,}", "despues del filtro"),
        metric_card("Edad media", f"{filtered['age'].mean():.1f}", "anos"),
        metric_card("Colesterol medio", f"{filtered['chol'].mean():.0f}", "mg/dL"),
        metric_card("Con enfermedad", f"{disease_rate:.1f}%", "num > 0"),
    ]


def model_unavailable() -> html.Div:
    return html.Div(
        [
            html.H3("Random Forest no disponible"),
            html.P(
                "La app esta lista para entrenarlo, pero falta instalar scikit-learn "
                "en este entorno. Ejecuta: pip install -r requirements.txt"
            ),
        ],
        className="model-message warning",
    )


def build_model(test_size: float, n_estimators: int, max_depth_value: str, mode: str):
    if not SKLEARN_AVAILABLE:
        return None

    target = (df["num"] > 0).astype(int) if mode == "binary" else df["num"].astype(int)
    max_depth = None if max_depth_value == "none" else int(max_depth_value)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return model, x_test, y_test, y_pred


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P("Dashboard clinico exploratorio", className="eyebrow"),
                        html.H1("Enfermedad cardiaca: analisis a pascientes"),
                        html.P(
                            "Explora el conjunto de datos, y en base a la estadística "
                            "prueba el riesgo de enfermedad cardíaca."
                        ),
                    ],
                    className="hero-copy",


                ),
                html.Div(id="summary-cards", className="metrics-grid"),
            ],
            className="hero",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.Div(
                            [
                                html.Label("Rango de edad"),
                                dcc.RangeSlider(
                                    id="age-range",
                                    min=int(df["age"].min()),
                                    max=int(df["age"].max()),
                                    step=1,
                                    value=[int(df["age"].min()), int(df["age"].max())],
                                    marks={30: "30", 45: "45", 60: "60", 75: "75"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                            className="control-block wide",
                        ),
                        html.Div(
                            [
                                html.Label("Objetivo visual"),
                                dcc.RadioItems(
                                    id="target-mode",
                                    options=[
                                        {"label": "Binario", "value": "binary"},
                                        {"label": "Severidad", "value": "severity"},
                                    ],
                                    value="binary",
                                    inline=True,
                                    className="segmented",
                                ),
                            ],
                            className="control-block",
                        ),

                    ],
                    className="controls-band",
                ),
                html.Section(
                    [

                        html.Div(
                            dcc.Graph(id="age-distribution"),
                            className="chart-full"
                        ),

                        html.Div(
                            [
                                dcc.Graph(id="numeric-box"),

                        html.Div(
                                    [
                                        html.Label("Variable numerica"),
                                        dcc.Dropdown(
                                            id="numeric-column",
                                            options=[
                                                {"label": label_for(c), "value": c}
                                                for c in NUMERIC_COLUMNS
                                            ],
                                            value="chol",
                                            clearable=False,
                                        ),
                                    ],
                                    className="chart-selector",
                                ),
                            ],
                            className="chart-card",
                        ),

                        html.Div(
                            [
                                dcc.Graph(id="category-bars"),

                                html.Div(
                                    [
                                        html.Label("Variable categorica"),
                                        dcc.Dropdown(
                                            id="category-column",
                                            options=[
                                                {"label": label_for(c), "value": c}
                                                for c in CATEGORICAL_COLUMNS
                                            ],
                                            value="cp",
                                            clearable=False,
                                        ),
                                    ],
                                    className="chart-selector",
                                ),
                            ],
                            className="chart-card",
                        ),

                        dcc.Graph(id="correlation-heatmap"),

                        dcc.Graph(id="normality-chart"),

                    ],
                    className="charts-grid",
                ),

                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Modelo predictivo", className="eyebrow"),
                                html.H2("Random Forest"),
                                html.P(
                                    "Sugerencia: usar primero clasificacion binaria para detectar presencia "
                                    "de enfermedad, porque el dataset tiene pocas muestras en severidades altas."
                                ),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Arboles"),
                                        dcc.Slider(
                                            id="trees-slider",
                                            min=50,
                                            max=400,
                                            step=50,
                                            value=200,
                                            marks={50: "50", 200: "200", 400: "400"},
                                        ),
                                    ],
                                    className="control-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Tamano de prueba"),
                                        dcc.Slider(
                                            id="test-size-slider",
                                            min=0.15,
                                            max=0.4,
                                            step=0.05,
                                            value=0.25,
                                            marks={0.15: "15%", 0.25: "25%", 0.4: "40%"},
                                        ),
                                    ],
                                    className="control-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Profundidad maxima"),
                                        dcc.Dropdown(
                                            id="max-depth",
                                            options=[
                                                {"label": "Sin limite", "value": "none"},
                                                {"label": "3", "value": "3"},
                                                {"label": "5", "value": "5"},
                                                {"label": "8", "value": "8"},
                                                {"label": "12", "value": "12"},
                                            ],
                                            value="5",
                                            clearable=False,
                                        ),
                                    ],
                                    className="control-block",
                                ),
                            ],
                            className="model-controls",
                        ),
                        html.Div(
                            [
                                html.Div(id="model-report", className="model-report"),
                                dcc.Graph(id="feature-importance"),
                                dcc.Graph(id="confusion-matrix"),
                            ],
                            className="model-grid",
                        ),
                    ],
                    className="model-band",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Prediccion individual", className="eyebrow"),
                                html.H2("Probar un paciente"),
                            ],
                            className="section-heading compact",
                        ),
                        html.Div(
                            [
                                html.Div([html.Label("Edad"), dcc.Input(id="p-age", type="number", value=55, min=18, max=100)]),
                                html.Div([html.Label("Presion reposo"), dcc.Input(id="p-trestbps", type="number", value=130)]),
                                html.Div([html.Label("Colesterol"), dcc.Input(id="p-chol", type="number", value=245)]),
                                html.Div([html.Label("FC maxima"), dcc.Input(id="p-thalach", type="number", value=150)]),
                                html.Div([html.Label("Oldpeak"), dcc.Input(id="p-oldpeak", type="number", value=1.0, step=0.1)]),
                                html.Div([html.Label("Sexo"), dcc.Dropdown(id="p-sex", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["sex"].items()], value=1, clearable=False)]),
                                html.Div([html.Label("Dolor pecho"), dcc.Dropdown(id="p-cp", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["cp"].items()], value=4, clearable=False)]),
                                html.Div([html.Label("Glucosa ayunas"), dcc.Dropdown(id="p-fbs", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["fbs"].items()], value=0, clearable=False)]),
                                html.Div([html.Label("ECG"), dcc.Dropdown(id="p-restecg", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["restecg"].items()], value=0, clearable=False)]),
                                html.Div([html.Label("Angina ejercicio"), dcc.Dropdown(id="p-exang", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["exang"].items()], value=0, clearable=False)]),
                                html.Div([html.Label("Pendiente ST"), dcc.Dropdown(id="p-slope", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["slope"].items()], value=2, clearable=False)]),
                                html.Div([html.Label("Vasos"), dcc.Dropdown(id="p-ca", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["ca"].items()], value=0, clearable=False)]),
                                html.Div([html.Label("Thal"), dcc.Dropdown(id="p-thal", options=[{"label": v, "value": k} for k, v in CATEGORY_LABELS["thal"].items()], value=3, clearable=False)]),
                            ],
                            className="patient-grid",
                        ),
                        html.Button("Calcular riesgo", id="predict-button", n_clicks=0, className="primary-button"),
                        html.Div(id="prediction-output", className="prediction-output"),
                    ],
                    className="predict-band",
                ),
            ]
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("summary-cards", "children"),
    Output("age-distribution", "figure"),
    Output("numeric-box", "figure"),
    Output("category-bars", "figure"),
    Output("correlation-heatmap", "figure"),
    Output("normality-chart", "figure"),
    Input("age-range", "value"),
    Input("target-mode", "value"),
    Input("numeric-column", "value"),
    Input("category-column", "value"),
)
def update_exploration(age_range, mode, numeric_column, category_column):
    filtered = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])].copy()
    target = target_column(mode)
    target_label = "Severidad" if mode == "severity" else "Diagnostico"

    if filtered.empty:
        blank = empty_figure("No hay registros para el filtro seleccionado")
        return make_summary_cards(filtered), blank, blank, blank, blank, blank

    age_fig = px.histogram(
        filtered,
        x="age",
        color=target,
        nbins=18,
        barmode="overlay",
        marginal="box",
        labels={"age": "Edad", target: target_label},
        title="Distribucion de edad por objetivo",
        template="plotly_white",
    )
    age_fig.update_traces(opacity=0.78)

    box_fig = px.box(
        filtered,
        x=target,
        y=numeric_column,
        color=target,
        points="suspectedoutliers",
        labels={target: target_label, numeric_column: label_for(numeric_column)},
        title=f"Distribucion de {label_for(numeric_column).lower()} por objetivo",
        template="plotly_white",
    )

    category_label_column = f"{category_column}_label"
    grouped = (
        filtered.groupby([category_label_column, target], observed=False)
        .size()
        .reset_index(name="Registros")
    )
    bar_fig = px.bar(
        grouped,
        x=category_label_column,
        y="Registros",
        color=target,
        barmode="group",
        labels={category_label_column: label_for(category_column), target: target_label},
        title=f"Agrupacion por {label_for(category_column).lower()}",
        template="plotly_white",
    )

    corr = filtered[FEATURE_COLUMNS + ["num"]].corr(numeric_only=True)
    heatmap_fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlacion entre variables clinicas",
        labels={"color": "r"},
        template="plotly_white",
    )
    heatmap_fig.update_xaxes(tickangle=45)

    stats = []
    for column in NUMERIC_COLUMNS:
        series = filtered[column].dropna()
        stats.append(
            {
                "Variable": label_for(column),
                "Asimetria": series.skew(),
                "Curtosis": series.kurt(),
            }
        )
    normality = pd.DataFrame(stats)
    normal_fig = go.Figure()
    normal_fig.add_bar(x=normality["Variable"], y=normality["Asimetria"], name="Asimetria")
    normal_fig.add_bar(x=normality["Variable"], y=normality["Curtosis"], name="Curtosis")
    normal_fig.add_hline(y=0, line_color="#394150", line_width=1)
    normal_fig.update_layout(
        title="Senales de normalidad: asimetria y curtosis",
        barmode="group",
        template="plotly_white",
        yaxis_title="Valor",
        legend_title="Estadistico",
    )

    for fig in [age_fig, box_fig, bar_fig, heatmap_fig, normal_fig]:
        fig.update_layout(margin={"l": 48, "r": 24, "t": 70, "b": 48}, title_x=0.02)

    return make_summary_cards(filtered), age_fig, box_fig, bar_fig, heatmap_fig, normal_fig


@app.callback(
    Output("model-report", "children"),
    Output("feature-importance", "figure"),
    Output("confusion-matrix", "figure"),
    Input("trees-slider", "value"),
    Input("test-size-slider", "value"),
    Input("max-depth", "value"),
    Input("target-mode", "value"),
)
def update_model(n_estimators, test_size, max_depth_value, mode):
    if not SKLEARN_AVAILABLE:
        return model_unavailable(), empty_figure("Instala scikit-learn para ver importancias"), empty_figure(
            "Instala scikit-learn para ver la matriz"
        )

    model, x_test, y_test, y_pred = build_model(test_size, n_estimators, max_depth_value, mode)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]

    cards = html.Div(
        [
            metric_card("Exactitud", f"{accuracy:.3f}", "holdout"),
            metric_card("F1 macro", f"{macro_f1:.3f}", "balance entre clases"),
            metric_card("Muestras test", f"{len(x_test)}", f"{test_size:.0%} del dataset"),
        ],
        className="metrics-grid compact-metrics",
    )

    importance = pd.DataFrame(
        {"Variable": [label_for(c) for c in FEATURE_COLUMNS], "Importancia": model.feature_importances_}
    ).sort_values("Importancia", ascending=True)
    importance_fig = px.bar(
        importance,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables del Random Forest",
        template="plotly_white",
        color="Importancia",
        color_continuous_scale="Teal",
    )
    importance_fig.update_layout(margin={"l": 120, "r": 24, "t": 70, "b": 48}, title_x=0.02)

    labels = sorted(pd.Series(y_test).unique())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_fig = px.imshow(
        matrix,
        x=[str(x) for x in labels],
        y=[str(x) for x in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de confusion",
        labels={"x": "Prediccion", "y": "Real", "color": "Casos"},
        template="plotly_white",
    )
    matrix_fig.update_layout(margin={"l": 70, "r": 24, "t": 70, "b": 48}, title_x=0.02)

    return cards, importance_fig, matrix_fig


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("trees-slider", "value"),
    State("test-size-slider", "value"),
    State("max-depth", "value"),
    State("p-age", "value"),
    State("p-trestbps", "value"),
    State("p-chol", "value"),
    State("p-thalach", "value"),
    State("p-oldpeak", "value"),
    State("p-sex", "value"),
    State("p-cp", "value"),
    State("p-fbs", "value"),
    State("p-restecg", "value"),
    State("p-exang", "value"),
    State("p-slope", "value"),
    State("p-ca", "value"),
    State("p-thal", "value"),
)
def predict_patient(
    n_clicks,
    n_estimators,
    test_size,
    max_depth_value,
    age,
    trestbps,
    chol,
    thalach,
    oldpeak,
    sex,
    cp,
    fbs,
    restecg,
    exang,
    slope,
    ca,
    thal,
):
    if n_clicks == 0:
        return html.P("Completa los valores y calcula una prediccion binaria de riesgo.")
    if not SKLEARN_AVAILABLE:
        return model_unavailable()

    model_result = build_model(test_size, n_estimators, max_depth_value, "binary")
    model = model_result[0]
    patient = pd.DataFrame(
        [
            {
                "age": age,
                "trestbps": trestbps,
                "chol": chol,
                "thalach": thalach,
                "oldpeak": oldpeak,
                "sex": sex,
                "cp": cp,
                "fbs": fbs,
                "restecg": restecg,
                "exang": exang,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    probability = model.predict_proba(patient)[0][1]
    predicted = "Con enfermedad" if probability >= 0.5 else "Sin enfermedad"
    tone = "risk-high" if probability >= 0.5 else "risk-low"

    return html.Div(
        [
            html.Span("Resultado estimado", className="metric-title"),
            html.Strong(predicted, className=f"prediction-label {tone}"),
            html.Span(f"Probabilidad de enfermedad: {probability:.1%}", className="metric-note"),
        ],
        className="prediction-card",
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
