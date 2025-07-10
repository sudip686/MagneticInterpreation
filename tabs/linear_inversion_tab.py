# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/tabs/linear_inversion_tab.py

import json
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

# Import the computational and plotting functions from utils
from utils import run_linear_inversion, plot_linear_model, plot_linear_data, plot_tikhonov_curve


class LinearInversionTab:
    """
    Manages the layout and callbacks for the 1D Linear Inversion tab.
    """

    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def layout(self):
        """
        Defines the layout of the 1D linear inversion tab.
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("▶️ Run 1D Inversion", id='run-linear-inversion-button', n_clicks=0,
                               className="my-3", color="primary"),
                    html.Div(id='linear-inversion-status'),
                    html.Hr(),
                    dbc.Label("Beta (Regularization)"),
                    dcc.Slider(id='beta-slider', min=0, max=19, step=1, value=10, marks=None,
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
        ])


    def register_callbacks(self):
        """
        Registers all callbacks associated with the 1D linear inversion tab.
        """

        @self.app.callback(
            Output('linear-inversion-results-store', 'data'),
            Output('linear-inversion-status', 'children'),
            Output('beta-slider', 'max'),
            Input('run-linear-inversion-button', 'n_clicks'),
            [
                # States for all 1D parameters from the sidebar
                State('lin-noise-floor-input', 'value'), State('lin-noise-percent-input', 'value'),
                State('lin-M-input', 'value'), State('lin-N-input', 'value'),
                State('lin-p-input', 'value'), State('lin-q-input', 'value'),
                State('lin-beta-min-input', 'value'), State('lin-beta-max-input', 'value'),
                State('lin-n-beta-input', 'value'), State('lin-m-ref-val-input', 'value'),
                State('lin-alpha-s-input', 'value'), State('lin-alpha-x-input', 'value'),
                State('lin-add-noise-checkbox', 'value'),
                State('lin-m-background-input', 'value'), State('lin-m1-input', 'value'),
                State('lin-m1-center-input', 'value'), State('lin-m1-width-input', 'value'),
                State('lin-m2-input', 'value'), State('lin-m2-center-input', 'value'),
                State('lin-m2-sigma-input', 'value')
            ],
            prevent_initial_call=True
        )
        def run_1d_inversion(n_clicks, noise_floor, noise_percent, M, N, p, q, beta_min, beta_max, n_beta,
                             m_ref_val, alpha_s, alpha_x, add_noise, m_background, m1, m1_center,
                             m1_width, m2, m2_center, m2_sigma):
            params = {
                'noise_floor': noise_floor, 'noise_percent': noise_percent, 'M': M, 'N': N, 'p': p, 'q': q,
                'beta_min': beta_min, 'beta_max': beta_max, 'n_beta': n_beta, 'm_ref_val': m_ref_val,
                'alpha_s': alpha_s, 'alpha_x': alpha_x, 'add_noise': add_noise,
                'm_background': m_background, 'm1': m1, 'm1_center': m1_center, 'm1_width': m1_width,
                'm2': m2, 'm2_center': m2_center, 'm2_sigma': m2_sigma
            }
            try:
                results = run_linear_inversion(params)

                # --- KEY CHANGE: Robustly convert all numpy arrays to lists ---
                for key, val in results.items():
                    if isinstance(val, np.ndarray):
                        results[key] = val.tolist()
                    # This handles the list of numpy arrays for the 'models' key
                    elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
                        results[key] = [item.tolist() for item in val]
                # --- END OF KEY CHANGE ---

                # The slider max is the number of betas minus 1 (for 0-based indexing)
                slider_max = n_beta - 1
                return json.dumps(results), dbc.Alert("✅ 1D Inversion Complete!", color="success"), slider_max
            except Exception as e:
                import traceback
                return no_update, dbc.Alert(f"❌ Error during 1D inversion: {e}\n{traceback.format_exc()}", color="danger"), no_update

        @self.app.callback(
            Output('linear-model-graph', 'figure'),
            Output('linear-data-graph', 'figure'),
            Output('linear-tikhonov-graph', 'figure'),
            Input('linear-inversion-results-store', 'data'),
            Input('beta-slider', 'value')
        )
        def update_1d_graphs(json_results, selected_beta_index):
            if json_results is None:
                # Return empty figures if no results are available
                empty_fig = go.Figure().update_layout(
                    xaxis={'visible': False}, yaxis={'visible': False},
                    annotations=[{"text": "Run 1D Inversion to see results", "xref": "paper", "yref": "paper",
                                  "showarrow": False, "font": {"size": 16}}]
                )
                return empty_fig, empty_fig, empty_fig

            results = json.loads(json_results)

            # Ensure index is within bounds
            if selected_beta_index >= len(results['models']):
                selected_beta_index = len(results['models']) - 1

            # Extract data for plotting
            m_true = np.array(results['m_true'])
            m_rec = np.array(results['models'][selected_beta_index])
            d_obs = np.array(results['d_obs'])
            uncertainty = np.array(results['uncertainty'])
            G = np.array(results['G'])
            d_pred = G @ m_rec
            x = np.array(results['x'])
            m_ref_val = results['m_ref_val']
            phi_ds = results['phi_ds']
            phi_ms = results['phi_ms']
            target_misfit = results['N']

            # Generate figures
            model_fig = plot_linear_model(x, m_true, m_rec, m_ref_val)
            data_fig = plot_linear_data(d_obs, d_pred, uncertainty)
            tikhonov_fig = plot_tikhonov_curve(phi_ds, phi_ms, selected_beta_index, target_misfit)

            return model_fig, data_fig, tikhonov_fig