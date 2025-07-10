import base64
import io
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.linalg import solve
from scipy.sparse import diags, vstack
import plotly.graph_objects as go

# Corrected imports to use lowercase 'simpeg'
from simpeg.potential_fields import magnetics
from simpeg.utils import model_builder
from simpeg.utils.io_utils import download
from simpeg import (
    data,
    data_misfit,
    directives,
    inversion,
    maps,
    inverse_problem,
    optimization,
    regularization,
    survey,
    simulation as simpeg_simulation,  # aliased to avoid conflict
)
from discretize import TensorMesh
from discretize.utils import active_from_xyz, mesh_builder_xyz


# --- 3D Inversion Logic ---
# The following functions are updated based on the provided working snippet
# to ensure compatibility with your SimPEG environment.

def setup_simpeg_simulation(
        _df, x_col, y_col, z_col, val_col,
        inducing_field_strength, inclination, declination,
        core_cell_size_x, core_cell_size_y, core_cell_size_z,
        padding_x, padding_y, padding_z,
        n_cpu=None,  # n_cpu is not used in the simulation constructor to avoid errors
        memory_mode='ram'
):
    """
    Sets up the SimPEG magnetic survey and simulation objects using a robust,
    explicit initialization pattern based on the provided working code.
    """
    receiver_locations = _df[[x_col, y_col, z_col]].values
    dobs = _df[val_col].values
    components = ["tmi"]

    # Define the source and survey using positional arguments for robustness
    source_field = magnetics.sources.UniformBackgroundField(
        [magnetics.receivers.Point(receiver_locations, components=components)],
        amplitude=inducing_field_strength,
        inclination=inclination,
        declination=declination
    )
    survey_obj = magnetics.survey.Survey([source_field])

    # Build the mesh
    mesh = mesh_builder_xyz(
        receiver_locations,
        [core_cell_size_x, core_cell_size_y, core_cell_size_z],
        padding_distance=[[padding_x, padding_x], [padding_y, padding_y], [padding_z, padding_z]],
        mesh_type='TENSOR'
    )

    # Define the active cells (below the topography)
    active_cells = active_from_xyz(mesh, receiver_locations, grid_reference='N')
    n_active = int(active_cells.sum())
    if n_active == 0:
        raise ValueError("No active cells found below data points. Adjust data or mesh parameters.")

    model_map = maps.IdentityMap(nP=n_active)

    # Use the older, more robust keywords from your working snippet.
    simulation = magnetics.simulation.Simulation3DIntegral(
        mesh=mesh,
        survey=survey_obj,
        model_map=model_map,
        active_cells=active_cells,
        store_sensitivities=memory_mode
    )
    # This explicit assignment is good practice with older versions.
    simulation.chiMap = model_map

    # Define the data object with standard deviation
    standard_deviation = np.maximum(0.02 * np.abs(dobs), 0.001) + 2
    data_object = data.Data(survey=survey_obj, dobs=dobs, standard_deviation=standard_deviation)

    return simulation, data_object, n_active


def run_smooth_inversion(
        _simulation, _data_object, _n_active,
        alpha_s, alpha_x, alpha_y, alpha_z,
        reference_model=None
):
    """Runs a smooth (L2-norm) inversion using a robust directive-based setup."""
    dmis = data_misfit.L2DataMisfit(data=_data_object, simulation=_simulation)

    reg = regularization.WeightedLeastSquares(
        mesh=_simulation.mesh,
        active_cells=_simulation.active_cells,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    )

    # Handle the reference model from Euler deconvolution if provided
    if reference_model is not None and reference_model.shape[0] == _n_active:
        reg.reference_model = reference_model
        m0 = reference_model.copy()  # Start from the reference model
    else:
        # Default starting model and reference model
        m0 = np.ones(_n_active) * 1e-4
        reg.reference_model = np.zeros(_n_active)

    opt = optimization.ProjectedGNCG(maxIter=20, lower=0.0, upper=10.0, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Set up a more detailed directive list based on the working example
    target_misfit = directives.TargetMisfit(chifact=1)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
    update_jacobi = directives.UpdatePreconditioner()
    directiveList = [starting_beta, target_misfit, beta_schedule, save_iteration, update_jacobi]

    inv = inversion.BaseInversion(inv_prob, directiveList=directiveList)

    try:
        # Run the inversion
        recovered_model = inv.run(m0)
    except MemoryError as e:
        raise MemoryError(
            "A MemoryError occurred. The problem is too large for the available RAM. "
            "Please reduce the problem size by increasing the 'Cell Size' values in the UI."
        ) from e

    return recovered_model


def run_sparse_inversion(
        _simulation, _data_object, _n_active,
        alpha_s, alpha_x, alpha_y, alpha_z,
        p_s, p_x, p_y, p_z,
        reference_model=None
):
    """Runs a sparse (IRLS) inversion using a robust directive-based setup."""
    dmis = data_misfit.L2DataMisfit(data=_data_object, simulation=_simulation)

    reg = regularization.Sparse(
        mesh=_simulation.mesh,
        active_cells=_simulation.active_cells,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    )
    reg.norms = [p_s, p_x, p_y, p_z]

    # Handle the reference model from Euler deconvolution if provided
    if reference_model is not None and reference_model.shape[0] == _n_active:
        reg.reference_model = reference_model
        m0 = reference_model.copy()
    else:
        # Default starting model and reference model
        m0 = np.ones(_n_active) * 1e-4
        reg.reference_model = np.zeros(_n_active)

    opt = optimization.ProjectedGNCG(maxIter=100, lower=0.0, upper=10.0, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Set up a more detailed directive list for IRLS based on the working example
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1)
    update_jacobi = directives.UpdatePreconditioner()
    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4, max_irls_iterations=30, coolEpsFact=1.5,
        beta_tol=1e-2, chifact_target=1
    )
    directiveList = [sensitivity_weights, starting_beta, update_IRLS, update_jacobi]

    inv = inversion.BaseInversion(inv_prob, directiveList=directiveList)

    try:
        # Run the inversion
        recovered_model = inv.run(m0)
    except MemoryError as e:
        raise MemoryError(
            "A MemoryError occurred. The problem is too large for the available RAM. "
            "Please reduce the problem size by increasing the 'Cell Size' values in the UI."
        ) from e

    return recovered_model


# ==============================================================================
# EULER DECONVOLUTION FUNCTION
# ==============================================================================
def run_euler_deconvolution(df, x_col, y_col, val_col, structural_index, window_size):
    print(f"--> Starting Euler Deconvolution with SI={structural_index} and window={window_size}...")
    grid_res_x = (df[x_col].max() - df[x_col].min()) / 100
    grid_res_y = (df[y_col].max() - df[y_col].min()) / 100
    grid_res = np.mean([grid_res_x, grid_res_y])
    grid_x, grid_y = np.mgrid[
                     df[x_col].min():df[x_col].max():grid_res,
                     df[y_col].min():df[y_col].max():grid_res
                     ]
    grid_tmi = griddata(df[[x_col, y_col]].values, df[val_col].values, (grid_x, grid_y), method='cubic')
    grid_tmi[np.isnan(grid_tmi)] = np.nanmean(grid_tmi)
    dz, dx = np.gradient(grid_tmi, grid_res)
    dy, _ = np.gradient(grid_tmi.T, grid_res)
    dy = dy.T
    solutions = []
    w = window_size // 2
    for i in range(w, grid_x.shape[0] - w):
        for j in range(w, grid_x.shape[1] - w):
            win_x = grid_x[i - w:i + w, j - w:j + w].ravel()
            win_y = grid_y[i - w:i + w, j - w:j + w].ravel()
            win_dx = dx[i - w:i + w, j - w:j + w].ravel()
            win_dy = dy[i - w:i + w, j - w:j + w].ravel()
            win_dz = dz[i - w:i + w, j - w:j + w].ravel()
            win_tmi = grid_tmi[i - w:i + w, j - w:j + w].ravel()
            x_center, y_center = grid_x[i, j], grid_y[i, j]
            A = np.vstack([win_dx, win_dy, win_dz, np.full_like(win_dx, -structural_index)]).T
            b = win_x * win_dx + win_y * win_dy - structural_index * win_tmi
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                solutions.append([x_center, y_center, -sol[2]])
            except np.linalg.LinAlgError:
                continue
    if not solutions:
        return pd.DataFrame(columns=['X', 'Y', 'Z_depth'])
    return pd.DataFrame(solutions, columns=['X', 'Y', 'Z_depth'])

# --- START: LINEAR INVERSION LOGIC (UNCHANGED) ---
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
        Wd = diags(1 / (uncertainty + 1e-8), 0) # Add epsilon to avoid division by zero
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

# --- 3D PLOTTING FUNCTION ---

def plot_simpeg_slice(mesh_props, model, active_cells, direction='Z', location_index=None):
    h_temp = [np.array(h_dim) for h_dim in mesh_props['h']]
    x0_temp = np.array(mesh_props['x0'])
    mesh = TensorMesh(h_temp, x0=x0_temp)
    full_model = np.full(mesh.nC, np.nan)
    full_model[active_cells] = model
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    dim_map = {'X': 0, 'Y': 1, 'Z': 2}
    max_idx = mesh.shape_cells[dim_map[direction]] - 1
    if location_index is None:
        location_index = max_idx // 2
    location_index = int(min(max(location_index, 0), max_idx))
    mesh.plot_slice(
        full_model,
        ax=ax,
        normal=direction,
        ind=location_index,
        grid=True,
        pcolor_opts={"cmap": "viridis"}
    )
    ax.set_aspect('equal')
    slice_coord = mesh.cell_centers[location_index, dim_map[direction]]
    ax.set_title(f'Model Slice at {direction}={slice_coord:.1f}m (Index: {location_index})')
    plt.tight_layout()
    return fig


def fig_to_uri(fig):
    """
    Converts a matplotlib figure to a URI for display in Dash.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return "data:image/png;base64,{}".format(data)
