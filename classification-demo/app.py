import dash
from dash import html, dcc, Input, Output, callback_context
import pandas as pd
import numpy as np
from dash_reusable_components import NamedSlider, ResetButton
from figures import generate_roc_curve, serve_prediction_plot, confusion_matrix, generate_confusion_matrix_table
from classification import run_classification, predict_with_threshold

# Get data
data = pd.read_csv("titanic.csv")
features = ['Age', 'Fare']

# Set default parameters
DEFAULT_TEST_SIZE = 0.3
DEFAULT_THRESHOLD = 0.0

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Titanic Survival Classification Visualization", style={'text-align': 'center'}),
    html.H3("CS-150 - Author: Meghan Nugent",
            style={'text-align': 'center'}),

    html.Div([
        NamedSlider("Test Split Ratio", id="test-split-slider",
                    min=0.1, max=0.5, step=0.05, value=DEFAULT_TEST_SIZE,
                    tooltip={"placement": "bottom", "always_visible": True}),
        NamedSlider("Classification Threshold", id="threshold-slider",
                    min=-1, max=1, step=0.1, value=DEFAULT_THRESHOLD,
                    tooltip={"placement": "bottom", "always_visible": True}),
        ResetButton("Reset to Defaults", id="reset-button")
    ], style={'width': '50%', 'margin': '0 auto'}),

    html.Div([
        html.H3("ROC Curve"),
        dcc.Graph(id="roc-curve-graph")
    ], style={'width': '80%', 'margin': '20px auto'}),

    html.Div([
        html.H3("Confusion Matrix"),
        html.Div(id="confusion-matrix-table")
    ], style={'width': '50%', 'margin': '20px auto'}),

    html.Div([
        html.H3("Decision Boundary Visualization"),
        dcc.Graph(id="decision-boundary-graph")
    ], style={'width': '80%', 'margin': '20px auto'}),
])

@app.callback(
    [Output("roc-curve-graph", "figure"),
     Output("confusion-matrix-table", "children"),
     Output("decision-boundary-graph", "figure"),
     Output("test-split-slider", "value"),
     Output("threshold-slider", "value")],
    [Input("test-split-slider", "value"),
     Input("threshold-slider", "value"),
     Input("reset-button", "n_clicks")]
)
def update_outputs(test_split, threshold, reset_clicks):
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == "reset-button":
        test_split = DEFAULT_TEST_SIZE
        threshold = DEFAULT_THRESHOLD

    model, scaler, X_train, X_test, y_train, y_test, _ = run_classification(data, features, 'Survived', test_split)

    # Compute decision function scores for the test set
    y_scores = model.decision_function(X_test)

    # Generate ROC curve figure
    roc_fig = generate_roc_curve(y_test, y_scores)

    # Generate confusion matrix
    y_pred_adjusted = predict_with_threshold(y_scores, threshold)
    conf_table = generate_confusion_matrix_table(y_test, y_pred_adjusted)

    # Generate decision boundary visualization
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    mesh_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.decision_function(grid_points_scaled)

    decision_boundary_fig = serve_prediction_plot(
        model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
    )

    return roc_fig, conf_table, decision_boundary_fig, test_split, threshold


if __name__ == '__main__':
    app.run_server(debug=True)
