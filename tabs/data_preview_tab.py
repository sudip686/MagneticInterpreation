# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/tabs/data_preview_tab.py
import pandas as pd
import io
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State


class DataPreviewTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def layout(self):
        return dcc.Graph(id='data-preview-scatter')

    def register_callbacks(self):
        @self.app.callback(
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
            if not all(col in df.columns for col in [x_col, y_col, val_col]):
                return go.Figure().update_layout(title="Error: Mapped columns not found in processed data.")
            fig = go.Figure(data=go.Scatter(
                x=df[x_col], y=df[y_col], mode='markers',
                marker=dict(color=df[val_col], colorscale='Viridis', showscale=True, colorbar=dict(title=val_col))
            ))
            fig.update_layout(title=f"Data Value Distribution ({val_col})", xaxis_title=x_col, yaxis_title=y_col)
            return fig
