from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

def generate_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Chance'))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    return fig


def generate_confusion_matrix_table(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Actual (Died)', 'Actual (Survived)'], columns=['Predicted (Died)', 'Predicted (Survived)'])
    table = dash_table.DataTable(
        data=df_cm.reset_index().to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df_cm.reset_index().columns],
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'}
    )
    return table


def serve_prediction_plot(model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold):
    # Compute predictions
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_train, y_pred_train)
    test_score = metrics.accuracy_score(y_test, y_pred_test)

    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    diff = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscales
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0, "#20e6ff"],
    ]

    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - diff,
        zmax=scaled_threshold + diff,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False,
            type="constraint",
            operation="=",
            value=scaled_threshold
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    # Training data points
    trace2 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale)
    )

    # Test data points
    trace3 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale)
    )

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"}
    )

    figure = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
    return figure
