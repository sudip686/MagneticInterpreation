# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/main.py

import base64
import io
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

# Import tab classes from the 'tabs' directory
from tabs.data_preview_tab import DataPreviewTab
from tabs.visualization_tab import Visualization3DTab
from tabs.smooth_inversion_tab import SmoothInversionTab
from tabs.sparse_inversion_tab import SparseInversionTab
from tabs.linear_inversion_tab import LinearInversionTab

# --- APP INITIALIZATION ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    suppress_callback_exceptions=True,
    title="Geophysical Inversion Suite"
)
server = app.server


# --- UI HELPER FUNCTIONS ---
def build_sidebar():
    """
    Creates the sidebar layout with all controls organized in an accordion.
    """
    return dbc.Col([
        html.H2("Controls", className="display-6"),
        html.Hr(),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                          'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                          'textAlign': 'center', 'margin': '10px 0'}),
                        html.Div(id='upload-status'),
                        dbc.Label("Skip Header Rows:"),
                        dcc.Input(id='skip-rows-input', type='number', value=0, min=0, step=1, className="mb-2"),
                        dbc.Label("X (Easting) Column:"),
                        dcc.Dropdown(id='x-col-dropdown'),
                        dbc.Label("Y (Northing) Column:", className="mt-2"),
                        dcc.Dropdown(id='y-col-dropdown'),
                        dbc.Label("Value Column (e.g., RTP):", className="mt-2"),
                        dcc.Dropdown(id='val-col-dropdown'),
                        dbc.Checkbox(id='use-constant-z-checkbox', label="Use a constant elevation (Z)", value=True,
                                     className="mt-3"),
                        dcc.Input(id='z-value-input', type='number', placeholder='Constant Elevation (m)', value=0.0),
                        dbc.Label("Z (Elevation) Column:", className="mt-2"),
                        dcc.Dropdown(id='z-col-dropdown', style={'display': 'none'}),
                        dbc.Button("Process & Load Data", id='load-data-button', n_clicks=0, className="mt-3 w-100",
                                   color="primary"),
                        html.Div(id='data-load-status', className="mt-2 text-success small")
                    ],
                    title="1. Data Loading & Mapping",
                ),
                dbc.AccordionItem(
                    [
                        dbc.Label("Inducing Field Strength (nT)"),
                        dcc.Input(id='inducing-field-strength-input', type='number', value=50000.0),
                        dbc.Label("Inclination (°)", className="mt-2"),
                        dcc.Input(id='inclination-input', type='number', value=90.0),
                        dbc.Label("Declination (°)", className="mt-2"),
                        dcc.Input(id='declination-input', type='number', value=0.0),
                        html.Hr(),

                        # --- START: UPDATED MESH CONTROLS ---
                        html.Div("Core Mesh Discretization", className="mt-3 fw-bold"),
                        dbc.Row([
                            dbc.Col(dbc.Label("Cell Size X (m)"), width=6),
                            dbc.Col(dbc.Label("Cell Size Y (m)"), width=6),
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Input(id='core-cell-size-x-input', type='number', value=50.0, min=1.0), width=6),
                            dbc.Col(dcc.Input(id='core-cell-size-y-input', type='number', value=50.0, min=1.0), width=6),
                        ]),
                        dbc.Label("Cell Size Z (m)", className="mt-2"),
                        dcc.Input(id='core-cell-size-z-input', type='number', value=50.0, min=1.0, className="mb-2"),

                        html.Div("Mesh Padding (m)", className="mt-3 fw-bold"),
                        dbc.Row([
                            dbc.Col(dbc.Label("X Padding"), width=6),
                            dbc.Col(dbc.Label("Y Padding"), width=6),
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Input(id='padding-x-input', type='number', value=200.0), width=6),
                            dbc.Col(dcc.Input(id='padding-y-input', type='number', value=200.0), width=6),
                        ]),
                        dbc.Label("Z Padding (Downwards)", className="mt-2"),
                        dcc.Input(id='padding-z-input', type='number', value=200.0, className="mb-2"),
                        # --- END: UPDATED MESH CONTROLS ---

                        html.Div("Alpha Values (Regularization Weights)", className="mt-3 fw-bold"),
                        dbc.Label("Alpha S (Smallness)"),
                        dcc.Slider(id='alpha-s-slider', min=0, max=10, step=0.1, value=1, marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        dbc.Label("Alpha X (Smoothness X)"),
                        dcc.Slider(id='alpha-x-slider', min=0, max=10, step=0.1, value=1, marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        dbc.Label("Alpha Y (Smoothness Y)"),
                        dcc.Slider(id='alpha-y-slider', min=0, max=10, step=0.1, value=1, marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        dbc.Label("Alpha Z (Smoothness Z)"),
                        dcc.Slider(id='alpha-z-slider', min=0, max=10, step=0.1, value=1, marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ],
                    title="2. 3D Inversion Parameters",
                ),
                dbc.AccordionItem(
                    [
                        html.Div("True Model Parameters (1D)", className="fw-bold"),
                        dbc.Label("Background"),
                        dcc.Input(id='lin-m-background-input', type='number', value=0.0),
                        dbc.Label("Anomaly 1 Value"),
                        dcc.Input(id='lin-m1-input', type='number', value=1.0),
                        dbc.Label("Anomaly 1 Center"),
                        dcc.Input(id='lin-m1-center-input', type='number', value=0.0),
                        dbc.Label("Anomaly 1 Width"),
                        dcc.Input(id='lin-m1-width-input', type='number', value=0.5),
                        dbc.Label("Anomaly 2 Value"),
                        dcc.Input(id='lin-m2-input', type='number', value=-0.5),
                        dbc.Label("Anomaly 2 Center"),
                        dcc.Input(id='lin-m2-center-input', type='number', value=1.0),
                        dbc.Label("Anomaly 2 Sigma"),
                        dcc.Input(id='lin-m2-sigma-input', type='number', value=0.2),
                        html.Hr(),
                        html.Div("Inversion Settings (1D)", className="fw-bold"),
                        dbc.Label("Noise Floor"),
                        dcc.Input(id='lin-noise-floor-input', type='number', value=0.01),
                        dbc.Label("Noise %"),
                        dcc.Input(id='lin-noise-percent-input', type='number', value=5),
                        dbc.Label("Model Cells (M)"),
                        dcc.Input(id='lin-M-input', type='number', value=100),
                        dbc.Label("Data Points (N)"),
                        dcc.Input(id='lin-N-input', type='number', value=20),
                        dbc.Label("Parameter p"),
                        dcc.Input(id='lin-p-input', type='number', value=-0.25, step=0.01),
                        dbc.Label("Parameter q"),
                        dcc.Input(id='lin-q-input', type='number', value=2.0, step=0.01),
                        dbc.Label("Beta Min (log10)"),
                        dcc.Input(id='lin-beta-min-input', type='number', value=-5),
                        dbc.Label("Beta Max (log10)"),
                        dcc.Input(id='lin-beta-max-input', type='number', value=2),
                        dbc.Label("Number of Betas"),
                        dcc.Input(id='lin-n-beta-input', type='number', value=20),
                        dbc.Label("Reference Model Value"),
                        dcc.Input(id='lin-m-ref-val-input', type='number', value=0.0),
                        dbc.Label("Alpha S"),
                        dcc.Input(id='lin-alpha-s-input', type='number', value=1.0),
                        dbc.Label("Alpha X"),
                        dcc.Input(id='lin-alpha-x-input', type='number', value=1.0),
                        dbc.Checkbox(id='lin-add-noise-checkbox', label="Add Noise to 1D Data", value=True,
                                     className="mt-2"),
                    ],
                    title="3. 1D Linear Inversion Parameters"
                )
            ],
            start_collapsed=False,
            always_open=True
        )
    ], md=3)


# --- INSTANTIATE TAB CLASSES ---
data_preview = DataPreviewTab(app)
vis_3d = Visualization3DTab(app)
smooth_inv = SmoothInversionTab(app)
sparse_inv = SparseInversionTab(app)
linear_inv = LinearInversionTab(app)

# --- MAIN APP LAYOUT ---
app.layout = dbc.Container([
    html.H1("🌍 Geophysical Inversion Suite", className="my-3"),
    html.Hr(),
    dbc.Row([
        build_sidebar(),
        dbc.Col([
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label="📄 Data Preview", children=data_preview.layout()),
                dcc.Tab(label="📊 3D Visualization", children=vis_3d.layout()),
                dcc.Tab(label="🔄 Smooth Inversion (L2)", children=smooth_inv.layout()),
                dcc.Tab(label="✨ Sparse Inversion (IRLS)", children=sparse_inv.layout()),
                dcc.Tab(label="📉 1D Linear Inversion", children=linear_inv.layout()),
            ])
        ], md=9)
    ]),
    # --- Data Stores for State Management ---
    dcc.Store(id='raw-data-store'),
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='column-store'),
    dcc.Store(id='smooth-inversion-results-store'),
    dcc.Store(id='sparse-inversion-results-store'),
    dcc.Store(id='linear-inversion-results-store'),
], fluid=True)


# --- CORE CALLBACKS (Managed by the main app) ---
@app.callback(
    Output('raw-data-store', 'data'),
    Output('column-store', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('skip-rows-input', 'value')
)
def handle_upload(contents, filename, skip_rows):
    if contents is None:
        return no_update, no_update, "Upload a CSV or TXT file to begin."
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        first_line = io.StringIO(decoded.decode('utf-8')).readline()
        sep = ',' if ',' in first_line else r'\s+'
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=sep, engine='python', skiprows=skip_rows or 0)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            # This loop is for basic validation, but the main processing happens in the next callback
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("No valid numerical data found after processing.")
        return df.to_json(orient='split'), df.columns.tolist(), dbc.Alert(f"✅ Loaded {filename}", color="info")
    except Exception as e:
        return None, None, dbc.Alert(f"❌ Error processing file: {e}", color="danger")


@app.callback(
    Output('processed-data-store', 'data'),
    Output('data-load-status', 'children'),
    Input('load-data-button', 'n_clicks'),
    State('raw-data-store', 'data'),
    State('x-col-dropdown', 'value'), State('y-col-dropdown', 'value'),
    State('z-col-dropdown', 'value'), State('val-col-dropdown', 'value'),
    State('use-constant-z-checkbox', 'value'), State('z-value-input', 'value'),
    prevent_initial_call=True
)
def process_and_load_data(n_clicks, json_data, x_col, y_col, z_col, val_col, use_constant_z, z_value):
    if not json_data:
        return no_update, dbc.Alert("Upload data first.", color="warning")
    if not all([x_col, y_col, val_col]):
        return no_update, dbc.Alert("Map all X, Y, and Value columns.", color="warning")

    df = pd.read_json(io.StringIO(json_data), orient='split')
    if use_constant_z:
        df['_generated_z_'] = z_value
    return df.to_json(orient='split'), f"Data processed with {len(df)} rows."


@app.callback(
    Output('x-col-dropdown', 'options'), Output('y-col-dropdown', 'options'),
    Output('val-col-dropdown', 'options'), Output('z-col-dropdown', 'options'),
    Output('x-col-dropdown', 'value'), Output('y-col-dropdown', 'value'),
    Output('val-col-dropdown', 'value'),
    Input('column-store', 'data')
)
def update_dropdowns(columns):
    if columns is None:
        return [[]] * 4 + [None] * 3
    options = [{'label': i, 'value': i} for i in columns]
    x_val = next((c for c in columns if 'x' in c.lower() or 'east' in c.lower()), columns[0] if columns else None)
    y_val = next((c for c in columns if 'y' in c.lower() or 'north' in c.lower()),
                 columns[1] if len(columns) > 1 else None)
    val_val = next((c for c in columns if 'val' in c.lower() or 'tmi' in c.lower() or 'rtp' in c.lower()),
                   columns[-1] if columns else None)
    return options, options, options, options, x_val, y_val, val_val


@app.callback(
    Output('z-col-dropdown', 'style'),
    Output('z-value-input', 'style'),
    Input('use-constant-z-checkbox', 'value')
)
def toggle_z_source(use_constant):
    if use_constant:
        return {'display': 'none'}, {'width': '100%'}
    else:
        return {'display': 'block', 'margin-top': '5px'}, {'display': 'none'}


# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)