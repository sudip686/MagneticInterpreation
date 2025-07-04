# C:/Users/SUDIPTA CHANDA/PycharmProjects/MagneticInterpreation/utils.py

import io
import base64
import numpy as np
import multiprocessing

# --- ESSENTIAL FIX for Matplotlib in a web server ---
# This must be done BEFORE importing pyplot. It tells Matplotlib to use a
# non-GUI backend, which is thread-safe and prevents the "main thread" error.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# --- END OF FIX ---

import plotly.graph_objects as go
from discretize import TensorMesh
from discretize.utils import mesh_builder_xyz, active_from_xyz
from scipy.sparse import diags, vstack

from simpeg.data import Data
from simpeg.data_misfit import L2DataMisfit
from simpeg.directives import (
    BetaEstimate_ByEig, TargetMisfit, BetaSchedule, SaveOutputEveryIteration,
    UpdatePreconditioner, UpdateSensitivityWeights, Update_IRLS
)
from simpeg.inverse_problem import BaseInvProblem
from simpeg.inversion import BaseInversion
from simpeg.maps import IdentityMap
from simpeg.optimization import ProjectedGNCG
from simpeg.regularization import WeightedLeastSquares, Sparse
from simpeg.potential_fields.magnetics.sources import UniformBackgroundField
from simpeg.potential_fields.magnetics.receivers import Point as MagPoint
from simpeg.potential_fields.magnetics.survey import Survey as MagSurvey
from simpeg.potential_fields.magnetics.simulation import Simulation3DIntegral


# --- Matplotlib to Dash Image Conversion ---
def fig_to_uri(in_fig, **save_args):
    """
    Saves a matplotlib figure to a base64 encoded URI for embedding in Dash.

    Args:
        in_fig (matplotlib.figure.Figure): The matplotlib figure to convert.
        **save_args: Additional keyword arguments to pass to fig.savefig().

    Returns:
        str: A base64 encoded URI string (e.g., "data:image/png;base64,...").
    """
    out_img = io.BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    plt.close(in_fig)  # Close the figure to free up memory
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii")
    return "data:image/png;base64,{}".format(encoded)


# --- 3D Inversion Logic ---
def setup_simpeg_simulation(
        _df, x_col, y_col, z_col, val_col,
        inducing_field_strength, inclination, declination,
        core_cell_size_x, core_cell_size_y, core_cell_size_z,
        padding_x, padding_y, padding_z
):
    """
    Sets up the SimPEG magnetic survey and simulation objects.
    """
    receiver_locations = _df[[x_col, y_col, z_col]].values
    dobs = _df[val_col].values
    components = ["tmi"]  # Total Magnetic Intensity

    # Define the uniform background inducing magnetic field
    source_field = UniformBackgroundField(receiver_list=[MagPoint(receiver_locations, components=components)],
                                          amplitude=inducing_field_strength, inclination=inclination,
                                          declination=declination)
    survey = MagSurvey(source_field)

    mesh = mesh_builder_xyz(
        receiver_locations,
        [core_cell_size_x, core_cell_size_y, core_cell_size_z],
        padding_distance=[
            [padding_x, padding_x],
            [padding_y, padding_y],
            [padding_z, padding_z]
        ],
        mesh_type='TENSOR'
    )

    active_cells = active_from_xyz(mesh, receiver_locations, grid_reference='N')

    if active_cells.sum() == 0:
        raise ValueError(
            "No active cells found. This might be due to data points being outside the mesh, or too small core cell size. Adjust data or mesh parameters.")

    n_active = int(active_cells.sum())
    model_map = IdentityMap(nP=n_active)

    # More conservative memory management
    sens_storage_option = "ram" if n_active < 50_000 else "disk"

    # --- CORRECTED SIMULATION CALL ---
    # The 'n_cpu' argument is not a valid keyword for this constructor, so it is removed.
    # The 'store_sensitivities' argument is correct and remains.
    simulation = Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        model_map=model_map,
        active_cells=active_cells,
        store_sensitivities=sens_storage_option
    )
    # --- END OF CORRECTION ---

    simulation.chiMap = model_map
    simulation.model = np.zeros(n_active)

    standard_deviation = np.maximum(0.02 * np.abs(dobs), 0.001) + 2
    data_object = Data(survey, dobs=dobs, standard_deviation=standard_deviation)

    return simulation, data_object, n_active


def run_smooth_inversion(_simulation, _data_object, _n_active, alpha_s, alpha_x, alpha_y, alpha_z):
    """
    Runs a smooth (L2-norm) inversion using SimPEG.
    """
    dmis = L2DataMisfit(data=_data_object, simulation=_simulation)

    reg = WeightedLeastSquares(
        mesh=_simulation.mesh, active_cells=_simulation.active_cells,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    )
    reg.reference_model = np.zeros(_n_active)

    opt = ProjectedGNCG(maxIter=20, lower=0.0, upper=10.0, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = BaseInvProblem(dmis, reg, opt)
    m0 = np.ones(_n_active) * 1e-4
    inv_prob.model = m0

    _simulation.model = m0
    _simulation.chiMap = _simulation.model_map

    target_misfit = TargetMisfit(target=_data_object.survey.nD)
    starting_beta = BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = BetaSchedule(coolingFactor=5, coolingRate=2)
    save_iteration = SaveOutputEveryIteration(save_txt=False)
    update_jacobi = UpdatePreconditioner()

    directiveList = [starting_beta, target_misfit, beta_schedule, save_iteration, update_jacobi]

    inv = BaseInversion(inv_prob, directiveList=directiveList)

    try:
        recovered_model = inv.run(m0)
    except MemoryError as e:
        raise MemoryError(
            "A MemoryError occurred during the smooth inversion. The problem is too large for the available RAM. Please reduce the problem size by increasing the 'Cell Size' values in the UI.") from e

    return recovered_model


def run_sparse_inversion(_simulation, _data_object, _n_active, alpha_s, alpha_x, alpha_y, alpha_z, p_s, p_x, p_y, p_z):
    """
    Runs a sparse (IRLS) inversion using SimPEG.
    """
    dmis = L2DataMisfit(data=_data_object, simulation=_simulation)

    reg = Sparse(
        mesh=_simulation.mesh, active_cells=_simulation.active_cells,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    )
    reg.reference_model = np.zeros(_n_active)
    reg.norms = [p_s, p_x, p_y, p_z]

    opt = ProjectedGNCG(maxIter=100, lower=0.0, upper=10.0, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = BaseInvProblem(dmis, reg, opt)
    m0 = np.ones(_n_active) * 1e-4
    inv_prob.model = m0

    _simulation.model = m0
    _simulation.chiMap = _simulation.model_map

    sensitivity_weights = UpdateSensitivityWeights(every_iteration=False)
    starting_beta = BetaEstimate_ByEig(beta0_ratio=1)
    update_jacobi = UpdatePreconditioner()
    update_IRLS = Update_IRLS(
        f_min_change=1e-4, max_irls_iterations=30, coolEpsFact=1.5,
        beta_tol=1e-2, chifact_target=1
    )
    directiveList = [sensitivity_weights, starting_beta, update_IRLS, update_jacobi]

    inv = BaseInversion(inv_prob, directiveList=directiveList)

    try:
        recovered_model = inv.run(m0)
    except MemoryError as e:
        raise MemoryError(
            "A MemoryError occurred during the sparse inversion. The problem is too large for the available RAM. Please reduce the problem size by increasing the 'Cell Size' values in the UI.") from e

    return recovered_model


def plot_simpeg_slice(mesh_props: dict, model: np.ndarray, active_cells: np.ndarray, slice_direction: str,
                      slice_location: int) -> plt.Figure:
    """
    Plots a slice of the recovered 3D SimPEG model.
    """
    h_temp = [np.array(h_dim) for h_dim in mesh_props['h']]
    x0_temp = np.array(mesh_props['x0'])
    mesh = TensorMesh(h_temp, x0=x0_temp)

    full_model = np.full(mesh.nC, np.nan)
    full_model[active_cells] = model

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if slice_direction == 'X':
        max_idx = mesh_props['vnC'][0] - 1
    elif slice_direction == 'Y':
        max_idx = mesh_props['vnC'][1] - 1
    else:  # Z
        max_idx = mesh_props['vnC'][2] - 1

    current_slice_location = int(min(max(slice_location, 0), max_idx))

    plot_obj = mesh.plot_slice(full_model, normal=slice_direction, ind=current_slice_location, ax=ax, grid=True,
                               pcolor_opts={"cmap": "viridis"})[0]
    cb = plt.colorbar(plot_obj, ax=ax)
    cb.set_label("Recovered Susceptibility (SI)")
    ax.set_aspect('equal', adjustable='box')

    title_map = {'X': f"East-West Slice (Index: {current_slice_location})",
                 'Y': f"North-South Slice (Index: {current_slice_location})",
                 'Z': f"Horizontal Slice (Index: {current_slice_location})"}
    ax.set_title(title_map[slice_direction])
    return fig


# --- START: LINEAR INVERSION LOGIC ---
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
        if not add_noise: return d_pred, np.zeros_like(d_pred)
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


def run_linear_inversion(params: dict) -> dict:
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
        'x': inv.x, 'betas': beta_values, 'phi_ds': [], 'phi_ms': [], 'models': [],
        'N': inv.N, 'm_ref_val': m_ref[0]
    }
    for beta in beta_values:
        m_rec, phi_d, phi_m = inv.run(
            G, d_obs, uncertainty, beta, params['alpha_s'], params['alpha_x'], m_ref
        )
        results['phi_ds'].append(phi_d)
        results['phi_ms'].append(phi_m)
        results['models'].append(m_rec)
    return results


def plot_linear_model(x: np.ndarray, m_true: np.ndarray, m_rec: np.ndarray, m_ref_val: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=m_true, mode='lines', name='True Model', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=x, y=m_rec, mode='lines', name='Recovered Model', line=dict(color='blue', width=2)))
    fig.add_hline(y=m_ref_val, line_dash="dash", line_color='grey', annotation_text='Reference Model')
    fig.update_layout(title="Model Space", xaxis_title="x", yaxis_title="m(x)", legend_title="Legend")
    return fig


def plot_linear_data(d_obs: np.ndarray, d_pred: np.ndarray, uncertainty: np.ndarray) -> go.Figure:
    data_index = np.arange(1, len(d_obs) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_index, y=d_obs, mode='markers', name='Observed Data',
                             error_y=dict(type='data', array=uncertainty, visible=True, color='black')))
    fig.add_trace(
        go.Scatter(x=data_index, y=d_pred, mode='lines', name='Predicted Data', line=dict(color='red', width=2)))
    fig.update_layout(title="Data Space", xaxis_title="Data Index", yaxis_title="d", legend_title="Legend")
    return fig


def plot_tikhonov_curve(phi_ds: list, phi_ms: list, selected_beta_index: int, target_misfit: float) -> go.Figure:
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

# --- END: LINEAR INVERSION LOGIC ---