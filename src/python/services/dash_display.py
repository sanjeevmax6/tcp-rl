from dash import Dash, html, dcc, Input, Output
import threading
import random
from . import globals  # adjust if you're not using relative imports

# -------------------- Color Map Generation --------------------
DISTINCT_COLORS = [
    "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#42d4f4", "#dcbeff"
]

def generate_color_map(test_cases):
    colors = {}
    for idx, val in enumerate(sorted(set(test_cases))):
        colors[val] = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
    return colors

# Precompute once
color_map = generate_color_map(range(100))  # up to 100 test cases

def render_colored_testcases(title, order, color_map, status_array=None, status_map=None):
    # Build the test case divs
    test_case_blocks = []
    for i, tc in enumerate(order):
        # Determine the border color from either array or map
        if status_array:
            status = status_array[i]
        elif status_map:
            status = status_map.get(tc, 0)
        else:
            status = None

        border_color = 'green' if status == 1 else 'red' if status == 0 else 'none'
        border_style = f"5px solid {border_color}" if border_color != 'none' else "none"

        test_case_blocks.append(
            html.Div(f"Test Case: {tc}", style={
                "backgroundColor": color_map.get(tc, "#666"),
                "color": "white",
                "padding": "8px",
                "margin": "6px 0",
                "borderRadius": "12px",
                "textAlign": "center",
                "fontSize": "16px",
                "width": "32%",
                "minWidth": "120px",
                "border": border_style
            })
        )

    # Wrap and return the column
    return html.Div([
        html.Div(title, style={"fontWeight": "bold", "marginBottom": "10px", "color": "white", "display": "flex", "flexDirection": "column", "alignItems": "center"}),
        html.Div(test_case_blocks, style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center"
        })
    ], style={"flex": "1", "margin": "10px"})

# -------------------- Dash Setup --------------------
app = Dash(__name__)
app.title = "RL Test Prioritization Visualizer"

def serve_layout():
    return html.Div([
        html.H2(f"Real Time Test Cases Reordering for {globals.subscript_env} Environment", style={
            "color": "white", "textAlign": "center", "marginBottom": "20px"
        }),
        html.Div(id="test-order-panel", style={
            "display": "flex",
            "flexDirection": "row",
            "gap": "60px",  # gap between columns
            "justifyContent": "center"
        }),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0)
    ], style={
        "backgroundColor": "#1e1e2f",
        "minHeight": "100vh",
        "padding": "30px",
        "fontFamily": "Arial, sans-serif"
    })

app.layout = serve_layout

# -------------------- Live Update Callback --------------------
@app.callback(
    Output("test-order-panel", "children"),
    Input("interval-component", "n_intervals")
)
def update_ui(n):
    all_test_cases = list(set(globals.original_order + globals.latest_order))
    color_map = generate_color_map(all_test_cases)
    status_map = {tc: status for tc, status in zip(globals.original_order, globals.failure_array)}

    return [
        render_colored_testcases("Original Order", globals.original_order, color_map, status_array=globals.failure_array),
        render_colored_testcases("Reordered", globals.latest_order, color_map, status_map=status_map)
    ]

def init_dashboard():
    thread = threading.Thread(target=app.run, kwargs={
        'debug': False, 'port': 8050, 'use_reloader': False
    })
    thread.daemon = True
    thread.start()
