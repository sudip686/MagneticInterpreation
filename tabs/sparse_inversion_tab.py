# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/tabs/sparse_inversion_tab.py
import json
import io
import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from utils import setup_simpeg_simulation, run_sparse_inversion, plot_simpeg_slice, fig_to_uri

# We need to reconstruct the mesh to get cell coordinates for the download
from discretize import TensorMesh
# NEW: Import cKDTree for efficiently mapping Euler solutions to the mesh
from scipy.spatial import cKDTree


class SparseInversionTab:
    """
    Manages the layout and callbacks for the Sparse (IRLS) Inversion tab.
    This version includes optimizations for speed and memory.
    """
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def layout(self):
        """
        Defines the layout of the sparse inversion tab.
        """
        return html.Div([
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "This inversion is slower but can produce sharper models. Use performance optimizations for best results.",
                ],
                color="info",
            ),
            dbc.Row([
                dbc.Col(dbc.Label("P-norm S (Smallness)"), width=3),
                dbc.Col(dcc.Slider(id='p-s-slider', min=0, max=2, step=0.1, value=0, marks={0: 'L0', 1: 'L1', 2: 'L2'},
                                   tooltip={"placement": "bottom", "always_visible": True}), width=9),
            ], className="mb-2 align-items-center"),
            dbc.Row([
                dbc.Col(dbc.Label("P-norm X (Smoothness X)"), width=3),
                dbc.Col(dcc.Slider(id='p-x-slider', min=0, max=2, step=0.1, value=1, marks={0: 'L0', 1: 'L1', 2: 'L2'},
                                   tooltip={"placement": "bottom", "always_visible": True}), width=9),
            ], className="mb-2 align-items-center"),
            dbc.Row([
                dbc.Col(dbc.Label("P-norm Y (Smoothness Y)"), width=3),
                dbc.Col(dcc.Slider(id='p-y-slider', min=0, max=2, step=0.1, value=1, marks={0: 'L0', 1: 'L1', 2: 'L2'},
                                   tooltip={"placement": "bottom", "always_visible": True}), width=9),
            ], className="mb-2 align-items-center"),
            dbc.Row([
                dbc.Col(dbc.Label("P-norm Z (Smoothness Z)"), width=3),
                dbc.Col(dcc.Slider(id='p-z-slider', min=0, max=2, step=0.1, value=1, marks={0: 'L0', 1: 'L1', 2: 'L2'},
                                   tooltip={"placement": "bottom", "always_visible": True}), width=9),
            ], className="mb-3 align-items-center"),

            dbc.Button("▶️ Run Sparse Inversion (IRLS)", id='run-sparse-inversion-button', n_clicks=0, className="my-2",
                       color="success"),

            dcc.Loading(id="loading-sparse", children=[
                html.Div(id='sparse-inversion-output'),
                html.Img(id='sparse-slice-graph', style={'width': '100%', 'max-width': '800px', 'margin': 'auto', 'display': 'block'})
            ]),

            html.Div([
                dbc.Button("💾 Download Model as CSV", id="download-sparse-model-button", color="secondary", className="mt-3"),
                dcc.Download(id="download-sparse-model-csv")
            ], id='sparse-download-controls', style={'display': 'none'}),

            html.Div([
                dbc.Label("Slice Direction", className="mt-3"),
                dbc.RadioItems(
                    id='sparse-slice-direction-radio',
                    options=[{'label': i, 'value': i} for i in ['X', 'Y', 'Z']],
                    value='Z',
                    inline=True
                ),
                dbc.Label("Slice Location", className="mt-2"),
                dcc.Slider(id='sparse-slice-location-slider', min=0, max=100, step=1, value=50,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], id='sparse-slice-controls', style={'display': 'none'}),
        ])

    def register_callbacks(self):
        """
        Registers all callbacks associated with the sparse inversion tab.
        """
        @self.app.callback(
            Output('sparse-inversion-results-store', 'data'),
            Output('sparse-inversion-output', 'children'),
            Output('sparse-download-controls', 'style'),
            Input('run-sparse-inversion-button', 'n_clicks'),
            [
                # Data and Inversion Parameters
                State('processed-data-store', 'data'),
                State('x-col-dropdown', 'value'), State('y-col-dropdown', 'value'),
                State('z-col-dropdown', 'value'), State('val-col-dropdown', 'value'),
                State('use-constant-z-checkbox', 'value'), State('z-value-input', 'value'),
                State('inducing-field-strength-input', 'value'), State('inclination-input', 'value'),
                State('declination-input', 'value'),
                State('core-cell-size-x-input', 'value'),
                State('core-cell-size-y-input', 'value'),
                State('core-cell-size-z-input', 'value'),
                State('padding-x-input', 'value'),
                State('padding-y-input', 'value'),
                State('padding-z-input', 'value'),
                State('alpha-s-slider', 'value'), State('alpha-x-slider', 'value'),
                State('alpha-y-slider', 'value'), State('alpha-z-slider', 'value'),
                State('p-s-slider', 'value'), State('p-x-slider', 'value'),
                State('p-y-slider', 'value'), State('p-z-slider', 'value'),
                # --- NEW: Read from optimization controls ---
                State('euler-results-store', 'data'),
                State('cpu-count-dropdown', 'value'),
                State('memory-mode-radio', 'value'),
            ],
            prevent_initial_call=True
        )
        def run_inversion(n_clicks, json_data, x_col, y_col, z_col, val_col, use_constant_z, z_value,
                          inducing_field, inclination, declination,
                          csx, csy, csz, pad_x, pad_y, pad_z,
                          alpha_s, alpha_x, alpha_y, alpha_z,
                          p_s, p_x, p_y, p_z,
                          # --- NEW: Accept optimization arguments ---
                          json_euler_results, n_cpu, memory_mode):

            if json_data is None:
                return no_update, dbc.Alert("Please upload and process data first.", color="warning"), {'display': 'none'}

            df = pd.read_json(io.StringIO(json_data), orient='split')
            z_col_to_use = '_generated_z_' if use_constant_z else z_col

            if not use_constant_z and z_col_to_use is None:
                return no_update, dbc.Alert("Please select a Z (Elevation) column when 'Use a constant elevation' is unchecked.", color="danger"), {'display': 'none'}

            try:
                # Pass the new optimization settings to the simulation setup
                sim, data_obj, n_active = setup_simpeg_simulation(
                    df, x_col, y_col, z_col_to_use, val_col,
                    float(inducing_field), float(inclination), float(declination),
                    float(csx), float(csy), float(csz),
                    float(pad_x), float(pad_y), float(pad_z),
                    n_cpu=n_cpu,
                    memory_mode=memory_mode
                )

                # --- NEW: Build Reference Model from Euler Results ---
                reference_model = None
                status_message = ""
                if json_euler_results:
                    euler_df = pd.read_json(io.StringIO(json_euler_results), orient='split')
                    if not euler_df.empty:
                        status_message = "--> Using Euler results to constrain inversion.\n"
                        reference_model = np.zeros(n_active)
                        active_cell_centers = sim.mesh.cell_centers[sim.active_cells]
                        tree = cKDTree(active_cell_centers)
                        ref_susceptibility = 0.01
                        dist, indices = tree.query(euler_df[['X', 'Y', 'Z_depth']].values, k=1)
                        reference_model[indices] = ref_susceptibility
                # --- END OF REFERENCE MODEL LOGIC ---

                recovered_model = run_sparse_inversion(
                    sim, data_obj, n_active,
                    float(alpha_s), float(alpha_x), float(alpha_y), float(alpha_z),
                    float(p_s), float(p_x), float(p_y), float(p_z),
                    reference_model=reference_model  # Pass the constrained model
                )

                results = {
                    'model': recovered_model.tolist(),
                    'active_cells': sim.active_cells.tolist(),
                    'mesh_props': {
                        'h': [h.tolist() for h in sim.mesh.h],
                        'x0': sim.mesh.x0.tolist(),
                        'vnC': list(sim.mesh.vnC)
                    }
                }
                success_alert = dbc.Alert(f"{status_message}✅ Sparse Inversion Complete!", color="success", style={'white-space': 'pre-wrap'})
                return json.dumps(results), success_alert, {'display': 'block'}
            except Exception as e:
                import traceback
                error_message = f"❌ An error occurred during inversion: {e}\n{traceback.format_exc()}"
                return None, dbc.Alert(error_message, color="danger", style={'white-space': 'pre-wrap'}), {'display': 'none'}

        # The update_slice and download_model_csv callbacks remain the same as before.
        # They are not shown here for brevity but should be included in your file.
        @self.app.callback(
            Output('sparse-slice-graph', 'src'),
            Output('sparse-slice-controls', 'style'),
            Output('sparse-slice-location-slider', 'max'),
            Output('sparse-slice-location-slider', 'value'),
            Input('sparse-inversion-results-store', 'data'),
            Input('sparse-slice-direction-radio', 'value'),
            Input('sparse-slice-location-slider', 'value')
        )
        def update_slice(json_results, direction, location):
            if json_results is None:
                return "", {'display': 'none'}, 100, 50

            results = json.loads(json_results)
            model = np.array(results['model'])
            active_cells = np.array(results['active_cells'], dtype=bool)
            mesh_props = results['mesh_props']

            if direction == 'X':
                max_val = mesh_props['vnC'][0] - 1
            elif direction == 'Y':
                max_val = mesh_props['vnC'][1] - 1
            else:  # Z
                max_val = mesh_props['vnC'][2] - 1

            current_location = int(min(max(location, 0), max_val))
            fig = plot_simpeg_slice(mesh_props, model, active_cells, direction, current_location)
            uri = fig_to_uri(fig)

            return uri, {'display': 'block'}, max_val, current_location

        @self.app.callback(
            Output('download-sparse-model-csv', 'data'),
            Input('download-sparse-model-button', 'n_clicks'),
            State('sparse-inversion-results-store', 'data'),
            prevent_initial_call=True,
        )
        def download_model_csv(n_clicks, json_results):
            if json_results is None:
                return no_update

            results = json.loads(json_results)
            model = np.array(results['model'])
            active_cells = np.array(results['active_cells'], dtype=bool)
            mesh_props = results['mesh_props']

            h_temp = [np.array(h_dim) for h_dim in mesh_props['h']]
            x0_temp = np.array(mesh_props['x0'])
            mesh = TensorMesh(h_temp, x0=x0_temp)

            active_cell_centers = mesh.cell_centers[active_cells]

            df_model = pd.DataFrame({
                'X': active_cell_centers[:, 0],
                'Y': active_cell_centers[:, 1],
                'Z': active_cell_centers[:, 2],
                'Susceptibility': model
            })

            return dcc.send_data_frame(df_model.to_csv, "sparse_inversion_model.csv", index=False)
