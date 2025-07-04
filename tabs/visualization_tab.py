# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/tabs/visualization_tab.py
import pandas as pd
import io
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State


class Visualization3DTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def layout(self):
        return dcc.Graph(id='3d-vis-graph', style={'height': '80vh'})

    def register_callbacks(self):
        @self.app.callback(
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
            required_3d_cols = [x_col, y_col, val_col, z_col_to_use]
            if not all(col in df.columns for col in required_3d_cols):
                return go.Figure().update_layout(title="Error: Mapped 3D columns not found in processed data.")
            fig = go.Figure(data=go.Scatter3d(
                x=df[x_col], y=df[y_col], z=df[z_col_to_use],
                mode='markers',
                marker=dict(size=4, color=df[val_col], colorscale='Viridis', showscale=True,
                            colorbar=dict(title=val_col))
            ))
            fig.update_layout(title="3D Data Scatter Plot",
                              scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title='Elevation'))
            return fig
