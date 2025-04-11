from dash import dcc, html

def NamedSlider(label, id, **kwargs):
    return html.Div(
        style={"padding": "10px 0"},
        children=[
            html.Label(label),
            dcc.Slider(id=id, **kwargs)
        ]
    )

def ResetButton(label, id):
    return html.Button(label, id=id, n_clicks=0)
