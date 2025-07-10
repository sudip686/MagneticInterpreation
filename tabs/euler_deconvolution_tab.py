import io
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from utils import run_euler_deconvolution


class EulerDeconvolutionTab:
    """
    Manages the layout and callbacks for the Euler Deconvolution tab.
    """
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def layout(self):
        """
        Defines the layout of the Euler Deconvolution tab.
        """
        return html.Div([
            # This component is not visible but enables file downloads
            dcc.Download(id="download-euler-csv"),

            dbc.Alert(
                [
                    html.I(className="bi bi-lightbulb-fill me-2"),
                    "This fast method estimates source locations. The results will automatically constrain the 3D inversions."
                ],
                color="primary"
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Structural Index (SI)"),
                    dcc.Slider(
                        id='euler-si-slider', min=0, max=3, step=0.1, value=1,
                        marks={0: '0 (Contact)', 1: '1 (Pipe)', 2: '2 (Sphere)', 3: '3'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Moving Window Size (cells)"),
                    dcc.Input(id='euler-window-input', type='number', value=10, min=3, step=1, className="w-100"),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(
                    dbc.Button("▶️ Run Euler Deconvolution", id='run-euler-button', n_clicks=0, className="w-100", color="info"),
                    width=8
                ),
                # --- NEW: Download Button ---
                dbc.Col(
                    dbc.Button("💾 Download CSV", id="btn-download-euler-csv", className="w-100", color="secondary"),
                    width=4
                ),
            ], className="my-2"),

            dcc.Loading(id="loading-euler", children=[
                html.Div(id='euler-output-status'),
                dcc.Graph(id='euler-results-graph')
            ]),
        ])

    def register_callbacks(self):
        """
        Registers all callbacks for the Euler Deconvolution tab.
        """
        # Callback to run the deconvolution
        @self.app.callback(
            Output('euler-results-store', 'data'),
            Output('euler-output-status', 'children'),
            Output('euler-results-graph', 'figure'),
            Input('run-euler-button', 'n_clicks'),
            [
                State('processed-data-store', 'data'),
                State('x-col-dropdown', 'value'),
                State('y-col-dropdown', 'value'),
                State('val-col-dropdown', 'value'),
                State('euler-si-slider', 'value'),
                State('euler-window-input', 'value'),
            ],
            prevent_initial_call=True
        )
        def run_euler(n_clicks, json_data, x_col, y_col, val_col, si, window):
            if not json_data:
                return no_update, dbc.Alert("Please upload and process data first.", color="warning"), {}

            if not all([x_col, y_col, val_col, si, window]):
                return no_update, dbc.Alert("Please ensure all columns and parameters are set.", color="warning"), {}

            df = pd.read_json(io.StringIO(json_data), orient='split')

            try:
                results_df = run_euler_deconvolution(df, x_col, y_col, val_col, si, window)

                if results_df.empty:
                    return no_update, dbc.Alert("Euler Deconvolution did not yield any solutions.", color="info"), {}

                # Create a plot of the results
                fig = px.scatter(
                    results_df,
                    x='X',
                    y='Y',
                    color='Z_depth',
                    title='Euler Deconvolution Solutions',
                    labels={'Z_depth': 'Depth (m)'},
                    color_continuous_scale=px.colors.sequential.Viridis_r
                )
                fig.update_layout(
                    yaxis_scaleanchor="x",
                    yaxis_scaleratio=1,
                )

                status = dbc.Alert(f"✅ Euler Deconvolution complete. Found {len(results_df)} solutions.", color="success")
                # Store results as JSON for other tabs to use
                return results_df.to_json(orient='split'), status, fig

            except Exception as e:
                import traceback
                error_message = f"❌ An error occurred during Euler Deconvolution: {e}\n{traceback.format_exc()}"
                return no_update, dbc.Alert(error_message, color="danger", style={'white-space': 'pre-wrap'}), {}

        # --- NEW: Callback to handle the download ---
        @self.app.callback(
            Output("download-euler-csv", "data"),
            Input("btn-download-euler-csv", "n_clicks"),
            State("euler-results-store", "data"),
            prevent_initial_call=True,
        )
        def download_csv(n_clicks, json_data):
            if not json_data:
                # Prevents download if no data is available
                return no_update

            # Convert the stored JSON back to a pandas DataFrame
            df = pd.read_json(io.StringIO(json_data), orient='split')

            # Use a Dash helper function to send the DataFrame as a CSV file
            return dcc.send_data_frame(df.to_csv, "euler_solutions.csv", index=False)
