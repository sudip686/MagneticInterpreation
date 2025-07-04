import base64
import io
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from discretize import TensorMesh
from discretize.utils import mesh_builder_xyz, active_from_xyz
from scipy.sparse import diags, vstack
# from matplotlib.ticker import ScalarFormatter # Not directly used in the provided code

# --- UNIFIED SIMPEG IMPORTS ---
from simpeg.data import Data
from simpeg.data_misfit import L2DataMisfit
from simpeg.directives import (
    BetaEstimate_ByEig,
    TargetMisfit,
    BetaSchedule,
    SaveOutputEveryIteration,
    UpdatePreconditioner,
    # UpdateSensitivityWeights, # Not used in provided code
    # SaveOutputDictEveryIteration, # Not used in provided code
    # Update_IRLS, # Not used in provided code
)
from simpeg.inverse_problem import BaseInvProblem
from simpeg.inversion import BaseInversion
from simpeg.maps import IdentityMap
from simpeg.optimization import ProjectedGNCG
from simpeg.regularization import WeightedLeastSquares  # Sparse was removed as it's not used
from simpeg.potential_fields.magnetics.sources import UniformBackgroundField
from simpeg.potential_fields.magnetics.receivers import Point as MagPoint
from simpeg.potential_fields.magnetics.survey import Survey as MagSurvey
from simpeg.potential_fields.magnetics.simulation import Simulation3DIntegral


# ======================================================================================
# HELPER FUNCTIONS AND CLASSES
# ======================================================================================

# --- Matplotlib to Dash Image Conversion ---
def fig_to_uri(in_fig, **save_args):
    """Saves a matplotlib figure to a URI."""
    out_img = io.BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    plt.close(in_fig)
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii")
    return "data:image/png;base64,{}".format(encoded)


# --- Linear Inversion ---
class LinearInversion:
    def __init__(self, M=100, N=20, p=-0.25, q=2.0):
        self.M, self.N, self.p, self.q = M, N, p, q
        self.x = np.linspace(-2, 2, M)

    def get_model(self, m_background, m1, m1_center, m1_width, m2, m2_center, m2_sigma):
        m = np.ones(self.M) * m_background
        m[np.abs(self.x - m1_center) < m1_width / 2.] = m1
        m += m2 * np.exp(-((self.x - m2_center) ** 2) / (2 * m2_sigma ** 2))
        return m

    def get_G(self):
        G = np.zeros((self.N, self.M))
        for i in range(self.N):
            G[i, :] = np.exp(self.p * i * self.x) * np.cos(2 * np.pi * self.q * i * self.x)
        return G

    def get_dobs(self, true_model, G, noise_floor, noise_percent, add_noise=True):
        d_pred = G @ true_model
        if not add_noise:
            return d_pred, np.zeros_like(d_pred)
        noise = noise_floor + (noise_percent / 100.) * np.abs(d_pred)
        np.random.seed(42)
        d_obs = d_pred + np.random.normal(0, 1, self.N) * noise
        return d_obs, noise

    def _get_W(self, alpha_s, alpha_x, m_ref):
        W_s = diags(np.ones(self.M), 0)
        W_x = diags([-1, 1], [0, 1], shape=(self.M - 1, self.M))
        W = vstack([np.sqrt(alpha_s) * W_s, np.sqrt(alpha_x) * W_x])
        w_m = np.concatenate([np.sqrt(alpha_s) * m_ref, np.zeros(self.M - 1)])
        return W, w_m

    def run(self, G, dobs, uncertainty, beta, alpha_s, alpha_x, m_ref):
        Wd = diags(1 / uncertainty, 0)
        G_w = Wd @ G
        d_w = Wd @ dobs
        W, w_m = self._get_W(alpha_s, alpha_x, m_ref)
        A = G_w.T @ G_w + beta * (W.T @ W)
        b = G_w.T @ d_w + beta * (W.T @ w_m)
        m_rec = np.linalg.solve(A, b)
        phi_d = np.linalg.norm(G_w @ m_rec - d_w) ** 2
        phi_m = np.linalg.norm(W @ m_rec - w_m) ** 2
        return m_rec, phi_d, phi_m


def run_linear_inversion(params):
    inv = LinearInversion(M=params['M'], N=params['N'], p=params['p'], q=params['q'])
    m_true = inv.get_model(
        params['m_background'], params['m1'], params['m1_center'], params['m1_width'],
        params['m2'], params['m2_center'], params['m2_sigma']
    )
    G = inv.get_G()
    d_obs, uncertainty = inv.get_dobs(
        m_true, G, params['noise_floor'], params['noise_percent'], params['add_noise']
    )
    beta_values = np.logspace(params['beta_min'], params['beta_max'], params['n_beta'])
    m_ref = np.ones(params['M']) * params['m_ref_val']
    results = {
        'm_true': m_true, 'd_obs': d_obs, 'uncertainty': uncertainty, 'G': G,
        'x': inv.x, 'betas': beta_values, 'phi_ds': [], 'phi_ms': [], 'models': []
    }
    for beta in beta_values:
        m_rec, phi_d, phi_m = inv.run(
            G, d_obs, uncertainty, beta, params['alpha_s'], params['alpha_x'], m_ref
        )
        results['phi_ds'].append(phi_d)
        results['phi_ms'].append(phi_m)
        results['models'].append(m_rec)
    return results


def plot_linear_model(x, m_true, m_rec, m_ref_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=m_true, mode='lines', name='True Model', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=x, y=m_rec, mode='lines', name='Recovered Model', line=dict(color='blue', width=2)))
    fig.add_hline(y=m_ref_val, line_dash="dash", line_color='grey', annotation_text='Reference Model')
    fig.update_layout(title="Model Space", xaxis_title="x", yaxis_title="m(x)", legend_title="Legend")
    return fig


def plot_linear_data(d_obs, d_pred, uncertainty):
    data_index = np.arange(1, len(d_obs) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_index, y=d_obs, mode='markers', name='Observed Data',
                             error_y=dict(type='data', array=uncertainty, visible=True, color='black')))
    fig.add_trace(
        go.Scatter(x=data_index, y=d_pred, mode='lines', name='Predicted Data', line=dict(color='red', width=2)))
    fig.update_layout(title="Data Space", xaxis_title="Data Index", yaxis_title="d", legend_title="Legend")
    return fig


def plot_tikhonov_curve(phi_ds, phi_ms, selected_beta_index, target_misfit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi_ms, y=phi_ds, mode='lines+markers', name='Tikhonov Curve', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=[phi_ms[selected_beta_index]], y=[phi_ds[selected_beta_index]], mode='markers',
                             name='Selected Beta', marker=dict(color='red', size=12, symbol='x')))
    fig.add_hline(y=target_misfit, line_dash="dash", line_color='grey',
                  annotation_text=f'Target Misfit (N={target_misfit})', annotation_position="bottom right")
    fig.update_layout(title="Tikhonov Curve (L-Curve)", xaxis_title="$\\phi_m$ (Model Norm)",
                      yaxis_title="$\\phi_d$ (Data Misfit)")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig


# --- SimPEG Inversion ---
def setup_simpeg_simulation(_df, x_col, y_col, z_col, val_col, inducing_field_strength, inclination, declination,
                            core_cell_size):
    receiver_locations = _df[[x_col, y_col, z_col]].values
    dobs = _df[val_col].values
    components = ["tmi"]
    source_field = UniformBackgroundField(receiver_list=[MagPoint(receiver_locations, components=components)],
                                          amplitude=inducing_field_strength, inclination=inclination,
                                          declination=declination)
    survey = MagSurvey(source_field)
    mesh = mesh_builder_xyz(receiver_locations, [core_cell_size, core_cell_size, core_cell_size],
                            padding_distance=[[200, 200], [200, 200], [200, 200]], mesh_type='TENSOR')
    active_cells = active_from_xyz(mesh, receiver_locations, grid_reference='N')

    # --- Robustness check for active cells ---
    if active_cells.sum() == 0:
        # Fallback: if no active cells, try a larger extent for active_from_xyz or raise error
        # For now, let's raise a more informative error.
        raise ValueError(
            "No active cells found. This might be due to data points being outside the mesh, or too small core cell size. Adjust data or mesh parameters.")

    n_active = int(active_cells.sum())
    model_map = IdentityMap(nP=n_active)
    sens_storage_option = "ram" if n_active < 500_000 else "disk"
    simulation = Simulation3DIntegral(survey=survey, mesh=mesh, model_map=model_map, active_cells=active_cells,
                                      store_sensitivities=sens_storage_option)
    # Ensure standard_deviation has a minimum value to prevent division by zero
    standard_deviation = np.maximum(0.02 * np.abs(dobs), 0.001) + 2  # Added a floor of 0.001
    data_object = Data(survey, dobs=dobs, standard_deviation=standard_deviation)
    return simulation, data_object, n_active


def run_simpeg_inversion(_simulation, _data_object, _n_active, alpha_s, alpha_x, alpha_y, alpha_z):
    dmis = L2DataMisfit(data=_data_object, simulation=_simulation)

    # The reference model is now set as a property after initialization, not during.
    reg = WeightedLeastSquares(
        mesh=_simulation.mesh,
        active_cells=_simulation.active_cells,
        alpha_s=alpha_s,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_z=alpha_z
    )
    reg.reference_model = np.zeros(_n_active)  # Set the reference model here
    # --- End of Key Change ---

    opt = ProjectedGNCG(maxIter=20, lower=0.0, upper=10.0, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = BaseInvProblem(dmis, reg, opt)

    target_misfit = TargetMisfit(target=_data_object.survey.nD)

    starting_beta = BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = BetaSchedule(coolingFactor=5, coolingRate=2)
    save_iteration = SaveOutputEveryIteration(save_txt=False)
    update_jacobi = UpdatePreconditioner()
    directiveList = [starting_beta, target_misfit, beta_schedule, save_iteration, update_jacobi]
    inv = BaseInversion(inv_prob, directiveList=directiveList)
    m0 = np.ones(_n_active) * 1e-4
    return inv.run(m0)


def plot_simpeg_slice(mesh_props: dict, model: np.ndarray, active_cells: np.ndarray, slice_direction: str,
                      slice_location: int) -> plt.Figure:
    # Reconstruct the mesh from stored properties
    # Ensure h elements are treated as lists of arrays
    h_temp = [np.array(h_dim) for h_dim in mesh_props['h']]
    x0_temp = np.array(mesh_props['x0'])
    mesh = TensorMesh(h_temp, x0=x0_temp)

    full_model = np.full(mesh.nC, np.nan)
    full_model[active_cells] = model
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Determine the maximum slice index for the given direction
    if slice_direction == 'X':
        max_idx = mesh.vnC[0] - 1
    elif slice_direction == 'Y':
        max_idx = mesh.vnC[1] - 1
    else:  # Z
        max_idx = mesh.vnC[2] - 1

    # Ensure slice_location is within valid bounds
    current_slice_location = int(min(max(slice_location, 0), max_idx))

    plot_obj = mesh.plot_slice(full_model, normal=slice_direction, ind=current_slice_location, ax=ax, grid=True,
                               pcolor_opts={"cmap": "viridis"})[0]
    cb = plt.colorbar(plot_obj, ax=ax)
    cb.set_label("Recovered Susceptibility (SI)")
    ax.set_aspect('equal', adjustable='box')
    # Include slice index in title for clarity
    title_map = {'X': f"East-West Slice (Index: {current_slice_location})",
                 'Y': f"North-South Slice (Index: {current_slice_location})",
                 'Z': f"Horizontal Slice (Index: {current_slice_location})"}
    ax.set_title(title_map[slice_direction])
    return fig


# ======================================================================================
# DASH APP LAYOUT
# ======================================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], suppress_callback_exceptions=True)
server = app.server


def control_card(title, controls, **kwargs):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(controls)], **kwargs)


sidebar = dbc.Col([
    html.H2("Controls", className="display-4"),
    html.Hr(),
    control_card("1. Data Loading & Mapping", [
        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'},
                   multiple=False),
        html.Div(id='upload-status'),
        dbc.Label("Skip Header Rows:"),  # Added label for clarity
        dcc.Input(id='skip-rows-input', type='number', placeholder='Header lines to skip', value=0, min=0, step=1,
                  style={'width': '100%', 'margin-bottom': '10px'}),
        dbc.Label("X (Easting) Column:"),
        dcc.Dropdown(id='x-col-dropdown', placeholder="Easting (X) Column"),
        dbc.Label("Y (Northing) Column:"),
        dcc.Dropdown(id='y-col-dropdown', placeholder="Northing (Y) Column", style={'margin-top': '5px'}),
        dbc.Label("Value Column (e.g., RTP):"),
        dcc.Dropdown(id='val-col-dropdown', placeholder="Value Column (e.g., RTP)", style={'margin-top': '5px'}),
        dbc.Checkbox(id='use-constant-z-checkbox', label="Use a constant elevation (Z)", value=True,
                     style={'margin-top': '10px'}),
        dcc.Input(id='z-value-input', type='number', placeholder='Constant Elevation (m)', value=0.0,
                  style={'width': '100%'}),
        dbc.Label("Z (Elevation) Column:"),
        dcc.Dropdown(id='z-col-dropdown', placeholder="Elevation (Z) Column",
                     style={'margin-top': '5px', 'display': 'none'}),
        dbc.Button("Process & Load Data", id='load-data-button', n_clicks=0, className="mt-3", color="primary"),
        html.Div(id='data-load-status', className="mt-2 text-success")
    ]),
    control_card("2. Magnetic Field Parameters", [
        dbc.Label("Inducing Field Strength (nT)"),
        dcc.Input(id='inducing-field-strength-input', type='number', value=50000.0, style={'width': '100%'}),
        dbc.Label("Inclination (°)"),
        dcc.Input(id='inclination-input', type='number', value=90.0, style={'width': '100%'}),
        dbc.Label("Declination (°)"),
        dcc.Input(id='declination-input', type='number', value=0.0, style={'width': '100%'}),
    ], className="mt-3"),
    control_card("3. Mesh & Regularization (3D)", [
        dbc.Label("Core Cell Size (m)"),
        dcc.Input(id='core-cell-size-input', type='number', value=50.0, min=1.0, style={'width': '100%'}),
        html.Div("Alpha Values (Regularization Weights)", className="mt-2"),
        dbc.Label("Alpha S (Smallness)"), dcc.Slider(id='alpha-s-slider', min=0, max=10, step=0.1, value=1),
        dbc.Label("Alpha X (Smoothness X)"), dcc.Slider(id='alpha-x-slider', min=0, max=10, step=0.1, value=1),
        dbc.Label("Alpha Y (Smoothness Y)"), dcc.Slider(id='alpha-y-slider', min=0, max=10, step=0.1, value=1),
        dbc.Label("Alpha Z (Smoothness Z)"), dcc.Slider(id='alpha-z-slider', min=0, max=10, step=0.1, value=1),
    ], className="mt-3"),
    control_card("4. 1D Inversion Parameters", [
        dbc.Label("Noise Floor"),
        dcc.Input(id='lin-noise-floor-input', type='number', value=0.01, style={'width': '100%'}),
        dbc.Label("Noise %"),
        dcc.Input(id='lin-noise-percent-input', type='number', value=5, style={'width': '100%'}),
        # Added 1D model parameters for full control
        dbc.Label("1D Model Cells (M)"),
        dcc.Input(id='lin-M-input', type='number', value=100, style={'width': '100%'}),
        dbc.Label("1D Data Points (N)"),
        dcc.Input(id='lin-N-input', type='number', value=20, style={'width': '100%'}),
        dbc.Label("1D Parameter p"),
        dcc.Input(id='lin-p-input', type='number', value=-0.25, step=0.01, style={'width': '100%'}),
        dbc.Label("1D Parameter q"),
        dcc.Input(id='lin-q-input', type='number', value=2.0, step=0.01, style={'width': '100%'}),
        dbc.Label("1D Beta Min (log10)"),
        dcc.Input(id='lin-beta-min-input', type='number', value=-5, style={'width': '100%'}),
        dbc.Label("1D Beta Max (log10)"),
        dcc.Input(id='lin-beta-max-input', type='number', value=2, style={'width': '100%'}),
        dbc.Label("1D Number of Betas"),
        dcc.Input(id='lin-n-beta-input', type='number', value=20, style={'width': '100%'}),
        dbc.Label("1D Reference Model Value"),
        dcc.Input(id='lin-m-ref-val-input', type='number', value=0.0, style={'width': '100%'}),
        dbc.Label("1D Alpha S"),
        dcc.Input(id='lin-alpha-s-input', type='number', value=1.0, style={'width': '100%'}),
        dbc.Label("1D Alpha X"),
        dcc.Input(id='lin-alpha-x-input', type='number', value=1.0, style={'width': '100%'}),
        dbc.Checkbox(id='lin-add-noise-checkbox', label="Add Noise to 1D Data", value=True,
                     style={'margin-top': '10px'}),
        html.Hr(),
        html.Div("True Model Parameters (1D)"),
        dbc.Label("Background"),
        dcc.Input(id='lin-m-background-input', type='number', value=0.0, style={'width': '100%'}),
        dbc.Label("Anomaly 1 Value"), dcc.Input(id='lin-m1-input', type='number', value=1.0, style={'width': '100%'}),
        dbc.Label("Anomaly 1 Center"),
        dcc.Input(id='lin-m1-center-input', type='number', value=0.0, style={'width': '100%'}),
        dbc.Label("Anomaly 1 Width"),
        dcc.Input(id='lin-m1-width-input', type='number', value=0.5, style={'width': '100%'}),
        dbc.Label("Anomaly 2 Value"), dcc.Input(id='lin-m2-input', type='number', value=-0.5, style={'width': '100%'}),
        dbc.Label("Anomaly 2 Center"),
        dcc.Input(id='lin-m2-center-input', type='number', value=1.0, style={'width': '100%'}),
        dbc.Label("Anomaly 2 Sigma"),
        dcc.Input(id='lin-m2-sigma-input', type='number', value=0.2, style={'width': '100%'}),
    ], className="mt-3")
], md=3)

content = dbc.Col([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label="📄 Data Preview", children=[dcc.Graph(id='data-preview-scatter')]),
        dcc.Tab(label="📊 3D Visualization", children=[dcc.Graph(id='3d-vis-graph', style={'height': '80vh'})]),
        dcc.Tab(label="🔄 Smooth Inversion (L2)", children=[
            dbc.Button("▶️ Run Smooth Inversion", id='run-smooth-inversion-button', n_clicks=0, className="my-2"),
            dcc.Loading(id="loading-smooth", children=[
                html.Div(id='smooth-inversion-output'),
                html.Img(id='smooth-slice-graph', style={'width': '100%'})
            ]),
            html.Div([
                dbc.RadioItems(id='smooth-slice-direction-radio',
                               options=[{'label': i, 'value': i} for i in ['X', 'Y', 'Z']], value='Z', inline=True),
                dcc.Slider(id='smooth-slice-location-slider', min=0, max=100, step=1, value=50)
                # Max will be updated by callback
            ], id='smooth-slice-controls', style={'display': 'none'}),  # Initially hidden
        ]),
        dcc.Tab(label="✨ Sparse Inversion (IRLS)", children=[
            html.Div("Sparse Inversion UI to be implemented.", className="p-4")
        ]),
        dcc.Tab(label="📉 1D Linear Inversion", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Button("▶️ Run 1D Inversion", id='run-linear-inversion-button', n_clicks=0,
                               className="my-3", color="primary"),
                    html.Div(id='linear-inversion-status'),
                    html.Hr(),
                    dbc.Label("Beta (Regularization)"),
                    dcc.Slider(id='beta-slider', min=0, max=19, step=1, value=10, marks=None,  # Max updated by callback
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=12)
            ], className="p-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='linear-model-graph'), md=6),
                dbc.Col(dcc.Graph(id='linear-data-graph'), md=6),
            ], className="p-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='linear-tikhonov-graph'), md=12)
            ], className="p-2")
        ]),
    ])
], md=9)

app.layout = dbc.Container([
    html.H1("🌍 GeoPhysical Data Interpreter (Dash Version)"),
    html.Hr(),
    dbc.Row([sidebar, content]),
    dcc.Store(id='raw-data-store'),
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='column-store'),
    dcc.Store(id='smooth-inversion-results-store'),
    dcc.Store(id='linear-inversion-results-store'),
], fluid=True)


# ======================================================================================
# DASH APP CALLBACKS
# ======================================================================================

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
        # Determine separator based on first line heuristic
        first_line_decoded = io.StringIO(decoded.decode('utf-8')).readline()
        sep = ',' if ',' in first_line_decoded else r'\s+'

        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')), sep=sep,
            engine='python', skiprows=skip_rows or 0)  # Use 0 if skip_rows is None

        # Clean column names (e.g., remove leading/trailing spaces from column headers)
        df.columns = df.columns.str.strip()

        for col in df.columns:
            # Try converting to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)  # Drop rows with any NaN values after numeric conversion

        if df.empty:
            raise ValueError("No valid numerical data found after processing. Check skip rows and data format.")

        return df.to_json(orient='split'), df.columns.tolist(), f"✅ Raw data loaded for {filename} ({len(df)} rows)."
    except Exception as e:
        import traceback
        return None, None, f"❌ Error processing file: {e}\n{traceback.format_exc()}"


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
        return no_update, "Upload data first."
    if not all([x_col, y_col, val_col]):
        return no_update, "Map all X, Y, and Value columns."

    df = pd.read_json(io.StringIO(json_data), orient='split')

    # Ensure selected columns exist in the DataFrame
    required_cols = [c for c in [x_col, y_col, val_col] if c is not None]
    if not all(col in df.columns for col in required_cols):
        return no_update, f"Error: Selected columns not found in data: {', '.join([c for c in required_cols if c not in df.columns])}"

    if use_constant_z:
        df['_generated_z_'] = float(z_value)  # Ensure z_value is treated as float
    elif z_col is None or z_col not in df.columns:
        return no_update, "Error: Z column not selected or not found in data when 'Use constant Z' is unchecked."

    # Return only the necessary columns for processed data to save space if DF is very large
    cols_to_save = [x_col, y_col, val_col]
    if use_constant_z:
        cols_to_save.append('_generated_z_')
    else:
        cols_to_save.append(z_col)

    # Handle potential non-numeric data if a user maps a non-numeric column by mistake
    for col in cols_to_save:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return no_update, f"Error: Column '{col}' is not numeric. Please check data or column mapping."

    return df[cols_to_save].to_json(orient='split'), f"Data processed and loaded with {len(df)} rows."


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
    # Attempt to pre-select common column names
    x_val = next((c for c in columns if 'x' in c.lower() or 'east' in c.lower() or 'easting' in c.lower()),
                 columns[0] if columns else None)
    y_val = next((c for c in columns if 'y' in c.lower() or 'north' in c.lower() or 'northing' in c.lower()),
                 columns[1] if len(columns) > 1 else None)
    val_val = next(
        (c for c in columns if 'val' in c.lower() or 'tmi' in c.lower() or 'rtp' in c.lower() or 'data' in c.lower()),
        columns[-1] if columns else None)
    # Don't pre-select Z as constant Z is default
    z_val = None
    return options, options, options, options, x_val, y_val, val_val


@app.callback(
    Output('z-col-dropdown', 'style'),
    Output('z-value-input', 'style'),
    Input('use-constant-z-checkbox', 'value')
)
def toggle_z_source(use_constant):
    if use_constant:
        return {'display': 'none', 'margin-top': '5px'}, {'width': '100%'}
    else:
        return {'display': 'block', 'margin-top': '5px'}, {'display': 'none'}


@app.callback(
    Output('data-preview-scatter', 'figure'),
    Input('processed-data-store', 'data'),
    State('x-col-dropdown', 'value'),
    State('y-col-dropdown', 'value'),
    State('val-col-dropdown', 'value')
)
def update_data_preview(json_data, x_col, y_col, val_col):
    if json_data is None or not all([x_col, y_col, val_col]):
        return go.Figure().update_layout(title="Click 'Process & Load Data' to see preview")

    df = pd.read_json(io.StringIO(json_data), orient='split')

    # Check if mapped columns exist in the processed dataframe
    if not all(col in df.columns for col in [x_col, y_col, val_col]):
        return go.Figure().update_layout(title="Error: Mapped columns not found in processed data.")

    fig = go.Figure(data=go.Scatter(
        x=df[x_col], y=df[y_col], mode='markers',
        marker=dict(color=df[val_col], colorscale='Viridis', showscale=True, colorbar=dict(title=val_col))
    ))
    fig.update_layout(title=f"Data Value Distribution ({val_col})", xaxis_title=x_col, yaxis_title=y_col)
    return fig


@app.callback(
    Output('3d-vis-graph', 'figure'),
    Input('processed-data-store', 'data'),
    State('x-col-dropdown', 'value'), State('y-col-dropdown', 'value'),
    State('z-col-dropdown', 'value'), State('val-col-dropdown', 'value'),
    State('use-constant-z-checkbox', 'value'), State('z-value-input', 'value')
)
def update_3d_vis(json_data, x_col, y_col, z_col, val_col, use_constant_z, z_value):
    if json_data is None or not all([x_col, y_col, val_col]):
        return go.Figure().update_layout(title="Click 'Process & Load Data' to see 3D visualization")

    df = pd.read_json(io.StringIO(json_data), orient='split')
    z_col_to_use = '_generated_z_' if use_constant_z else z_col

    # Ensure all required columns are present after loading from store
    required_3d_cols = [x_col, y_col, val_col, z_col_to_use]
    if not all(col in df.columns for col in required_3d_cols):
        return go.Figure().update_layout(
            title="Error: Mapped 3D columns not found in processed data. Check Z settings.")

    fig = go.Figure(data=go.Scatter3d(
        x=df[x_col], y=df[y_col], z=df[z_col_to_use],  # FIXED: z_data was undefined
        mode='markers',
        marker=dict(size=4, color=df[val_col], colorscale='Viridis', showscale=True, colorbar=dict(title=val_col))
    ))
    fig.update_layout(title="3D Data Scatter Plot",
                      scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title='Elevation'))
    return fig


# --- Callback to run Smooth Inversion ---
@app.callback(
    Output('smooth-inversion-results-store', 'data'),
    Output('smooth-inversion-output', 'children'),
    Input('run-smooth-inversion-button', 'n_clicks'),
    State('processed-data-store', 'data'),  # FIXED: Changed 'data-store' to 'processed-data-store'
    State('x-col-dropdown', 'value'), State('y-col-dropdown', 'value'),
    State('z-col-dropdown', 'value'), State('val-col-dropdown', 'value'),
    State('use-constant-z-checkbox', 'value'), State('z-value-input', 'value'),
    State('inducing-field-strength-input', 'value'), State('inclination-input', 'value'),
    State('declination-input', 'value'), State('core-cell-size-input', 'value'),
    State('alpha-s-slider', 'value'), State('alpha-x-slider', 'value'),
    State('alpha-y-slider', 'value'), State('alpha-z-slider', 'value'),
    prevent_initial_call=True
)
def run_smooth_inversion(n_clicks, json_data, x_col, y_col, z_col, val_col, use_constant_z, z_value,
                         inducing_field, inclination, declination, core_cell,
                         alpha_s, alpha_x, alpha_y, alpha_z):
    if json_data is None:
        return None, "Please upload and process data first."  # Updated message

    df = pd.read_json(io.StringIO(json_data), orient='split')

    # Validate inputs before running complex simulation
    if not all([x_col, y_col, val_col, inducing_field is not None, inclination is not None, declination is not None,
                core_cell is not None,
                alpha_s is not None, alpha_x is not None, alpha_y is not None, alpha_z is not None]):
        return None, "Please ensure all required input fields are filled for Smooth Inversion."

    if use_constant_z:
        df['_generated_z_'] = float(z_value)
        z_col_to_use = '_generated_z_'
    else:
        z_col_to_use = z_col
        if z_col_to_use is None or z_col_to_use not in df.columns:
            return None, "Error: Z column not selected or not found in data when 'Use constant Z' is unchecked."

    # Final check on columns after potential z_col_to_use assignment
    required_sim_cols = [x_col, y_col, z_col_to_use, val_col]
    if not all(col in df.columns for col in required_sim_cols):
        return None, f"Error: Some required columns for simulation are missing: {', '.join([c for c in required_sim_cols if c not in df.columns])}"

    try:
        sim, data_obj, n_active = setup_simpeg_simulation(
            df, x_col, y_col, z_col_to_use, val_col,
            float(inducing_field), float(inclination), float(declination), float(core_cell)
        )
        recovered_model = run_simpeg_inversion(
            sim, data_obj, n_active, float(alpha_s), float(alpha_x), float(alpha_y), float(alpha_z)
        )

        # Prepare results for storage (must be JSON serializable)
        results = {
            'model': recovered_model.tolist(),
            'active_cells': sim.active_cells.tolist(),
            'mesh_props': {
                'h': [h_dim.tolist() for h_dim in sim.mesh.h],
                'x0': sim.mesh.x0.tolist(),
                'nC': sim.mesh.nC.tolist(),  # Store nC for recreating mesh correctly
                'vnC': sim.mesh.vnC.tolist()  # Store vnC for recreating mesh correctly
            }
        }
        return json.dumps(results), "✅ Smooth Inversion Complete!"
    except Exception as e:
        import traceback
        return None, f"❌ An error occurred during inversion: {e}\n{traceback.format_exc()}"


# --- Callback to update slice controls and plot ---
@app.callback(
    Output('smooth-slice-graph', 'src'),
    Output('smooth-slice-controls', 'style'),
    Output('smooth-slice-location-slider', 'max'),
    Output('smooth-slice-location-slider', 'value'),
    Input('smooth-inversion-results-store', 'data'),
    Input('smooth-slice-direction-radio', 'value'),
    Input('smooth-slice-location-slider', 'value')
)
def update_smooth_slice(json_results, direction, location):
    if json_results is None:
        return "", {'display': 'none'}, 100, 50

    results = json.loads(json_results)
    model = np.array(results['model'])
    active_cells = np.array(results['active_cells'], dtype=bool)
    mesh_props = results['mesh_props']

    # Determine max slider value based on direction
    # Use vnC from mesh_props which represents the number of cells in each direction
    if direction == 'X':
        max_val = mesh_props['vnC'][0] - 1
    elif direction == 'Y':
        max_val = mesh_props['vnC'][1] - 1
    else:  # Z
        max_val = mesh_props['vnC'][2] - 1

    # Ensure location is within bounds and convert to int
    current_location = int(min(max(location, 0), max_val))

    fig = plot_simpeg_slice(mesh_props, model, active_cells, direction, current_location)
    uri = fig_to_uri(fig)

    return uri, {'display': 'block'}, max_val, current_location


# --- Callback to run 1D Linear Inversion ---
@app.callback(
    Output('linear-inversion-results-store', 'data'),
    Output('linear-inversion-status', 'children'),
    Output('beta-slider', 'max'),
    Output('beta-slider', 'value'),
    Input('run-linear-inversion-button', 'n_clicks'),
    State('lin-M-input', 'value'), State('lin-N-input', 'value'),
    State('lin-p-input', 'value'), State('lin-q-input', 'value'),
    State('lin-noise-floor-input', 'value'), State('lin-noise-percent-input', 'value'),
    State('lin-beta-min-input', 'value'), State('lin-beta-max-input', 'value'),
    State('lin-n-beta-input', 'value'), State('lin-m-ref-val-input', 'value'),
    State('lin-alpha-s-input', 'value'), State('lin-alpha-x-input', 'value'),
    State('lin-add-noise-checkbox', 'value'),
    State('lin-m-background-input', 'value'), State('lin-m1-input', 'value'),
    State('lin-m1-center-input', 'value'), State('lin-m1-width-input', 'value'),
    State('lin-m2-input', 'value'), State('lin-m2-center-input', 'value'),
    State('lin-m2-sigma-input', 'value'),
    prevent_initial_call=True
)
def run_linear_inversion_callback(n_clicks, M, N, p, q, noise_floor, noise_percent,
                                  beta_min, beta_max, n_beta, m_ref_val, alpha_s, alpha_x,
                                  add_noise, m_background, m1, m1_center, m1_width,
                                  m2, m2_center, m2_sigma):
    # Validate all inputs
    params_values = [M, N, p, q, noise_floor, noise_percent, beta_min, beta_max, n_beta,
                     m_ref_val, alpha_s, alpha_x, m_background, m1, m1_center, m1_width,
                     m2, m2_center, m2_sigma]

    if not all(val is not None for val in params_values):
        return no_update, "Please fill all 1D Inversion parameters.", no_update, no_update

    params = {
        'M': M, 'N': N, 'p': p, 'q': q,
        'noise_floor': noise_floor, 'noise_percent': noise_percent, 'add_noise': add_noise,
        'beta_min': beta_min, 'beta_max': beta_max, 'n_beta': n_beta,
        'm_ref_val': m_ref_val, 'alpha_s': alpha_s, 'alpha_x': alpha_x,
        'm_background': m_background, 'm1': m1, 'm1_center': m1_center, 'm1_width': m1_width,
        'm2': m2, 'm2_center': m2_center, 'm2_sigma': m2_sigma
    }
    try:
        results = run_linear_inversion(params)
        # Store results as JSON serializable data
        serializable_results = {
            'm_true': results['m_true'].tolist(),
            'd_obs': results['d_obs'].tolist(),
            'uncertainty': results['uncertainty'].tolist(),
            'G': results['G'].tolist(),  # Store G as well for predicted data calculation
            'x': results['x'].tolist(),
            'betas': results['betas'].tolist(),
            'phi_ds': results['phi_ds'],
            'phi_ms': results['phi_ms'],
            'models': [m.tolist() for m in results['models']],
            'N': N  # Store N for target misfit calculation in plot
        }
        # Update beta slider max based on number of betas
        beta_slider_max = len(results['betas']) - 1
        return json.dumps(serializable_results), "✅ 1D Inversion Complete!", beta_slider_max, int(
            beta_slider_max / 2)  # Set default slider to middle
    except Exception as e:
        import traceback
        return None, f"❌ An error occurred during 1D inversion: {e}\n{traceback.format_exc()}", no_update, no_update


# --- Callbacks to update 1D plots based on beta slider ---
@app.callback(
    Output('linear-model-graph', 'figure'),
    Output('linear-data-graph', 'figure'),
    Output('linear-tikhonov-graph', 'figure'),
    Input('beta-slider', 'value'),
    State('linear-inversion-results-store', 'data')
)
def update_linear_plots(selected_beta_index, json_results):
    if json_results is None:
        # Default empty figures
        return go.Figure().update_layout(title="Run 1D Inversion to see model"), \
            go.Figure().update_layout(title="Run 1D Inversion to see data"), \
            go.Figure().update_layout(title="Run 1D Inversion to see Tikhonov Curve")

    results = json.loads(json_results)

    # Ensure selected_beta_index is within bounds of available models/betas
    if selected_beta_index >= len(results['models']):
        selected_beta_index = len(results['models']) - 1
    if selected_beta_index < 0:
        selected_beta_index = 0

    m_rec = np.array(results['models'][selected_beta_index])
    beta_selected = results['betas'][selected_beta_index]

    # Re-calculate predicted data for the selected model
    # Note: G is stored, so we can compute d_pred using the selected m_rec
    G_matrix = np.array(results['G'])  # Reconstruct G from the stored data
    d_pred_selected_model = G_matrix @ m_rec

    # Assuming N is the number of data points, target misfit is N
    # This should be consistent with how target_misfit is set in SimPEG, typically survey.nD
    # For 1D, N is directly available from params
    target_misfit = results['N'] if 'N' in results else len(results['d_obs'])  # Fallback if N not stored

    # Plotting functions
    model_fig = plot_linear_model(np.array(results['x']), np.array(results['m_true']), m_rec, results['m_ref_val'])
    data_fig = plot_linear_data(np.array(results['d_obs']), d_pred_selected_model, np.array(results['uncertainty']))
    tikhonov_fig = plot_tikhonov_curve(results['phi_ds'], results['phi_ms'], selected_beta_index, target_misfit)

    # Add beta value to model and data plot titles
    model_fig.update_layout(title=f"Model Space ($\\beta$ = {beta_selected:.2e})")
    data_fig.update_layout(title=f"Data Space ($\\beta$ = {beta_selected:.2e})")

    return model_fig, data_fig, tikhonov_fig


# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)