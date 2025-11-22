# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import numpy as onp
import jax
import jax.numpy as jnp
from jax import Array
import optax
from diffrax import Tsit5, Euler, Heun, Midpoint, Ralston, Bosh3, Dopri5, Dopri8, ImplicitEuler, Kvaerno3
from diffrax import ConstantStepSize, PIDController

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator

from pathlib import Path
from tqdm import tqdm
import time

from soromox.systems.system_state import SystemState
from soromox.systems.planar_pcs import PlanarPCS
from soromox.systems.planar_pcs_simplified import PlanarPCS_simple
from soromox.utils.lie_algebra.se2 import exp_SE2
from my_utilis import *

# Jax settings
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

# Folders
curr_folder = Path(__file__).parent
plots_folder = curr_folder/'plots and videos'/Path(__file__).stem
plots_folder.mkdir(parents=True, exist_ok=True)

data_folder = curr_folder/'saved data'/Path(__file__).stem
data_folder.mkdir(parents=True, exist_ok=True)

# Functions for plotting
def draw_robot(
    robot: PlanarPCS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(
        robot.forward_kinematics, in_axes=(None, 0), out_axes=-1
    )
    L_max = jnp.sum(robot.L)

    s_ps = jnp.linspace(0, L_max, num_points)
    chi_ps = batched_forward_kinematics(q, s_ps)

    curve = onp.array(chi_ps[1:, :], dtype=onp.float64).T  # (N,2)

    chi_point1 = robot.forward_kinematics(q, robot.L[0])
    pos_point1 = onp.array(chi_point1[1:], dtype=onp.float64).T
    pos_point2 = curve[-1]
    points_position = onp.stack([pos_point1, pos_point2])

    return curve, points_position

def animate_robot_matplotlib(
    robot: PlanarPCS,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    width = jnp.linalg.norm(robot.L) * 6
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Base
    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    if animation:
        (line,) = ax.plot([], [], lw=4, color="blue")
        (points,) = ax.plot([], [], 'ro', markersize=5)
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        ax.grid(True)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            points.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, points, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            draw_base(ax, robot, L=0.1)
            curve, tar_pos = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            points.set_data(tar_pos[:, 0], tar_pos[:, 1])
            title_text.set_text(f"t = {t:.2f} s")
            return line, title_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_list),
            init_func=init,
            blit=False,
            interval=interval,
        )

        if show:
            plt.show()
        plt.close(fig)

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            ax.grid(True)
            draw_base(ax, robot, L=0.1)
            q = q_list[frame_idx]
            curve, points_pos = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            ax.plot(points_pos[:, 0], points_pos[:, 1], 'ro', markersize=5)
            fig.canvas.draw_idle()

        # Create slider
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

        if show:
            plt.show()

        plt.close(fig)


# =====================================================
# Target creation
# =====================================================

# End effector target: x (for target1) and y (for target2) coordinates
t0 = 0.0
t1 = 5.0
dt_target = 1e-3
t_target = jnp.arange(t0, t1, dt_target)

c1 = 0.2
k1 = 29
wd1 = jnp.sqrt(4*k1-c1**2)/2
target1 = 0.06 * jnp.exp(-c1/2 * t_target) * jnp.cos(wd1 * t_target)
T_target1 = 2*jnp.pi/wd1

c2 = 0.2
k2 = 30 #### 30 ####
wd2 = jnp.sqrt(4*k2-c2**2)/2
target2 = 0.015 * jnp.exp(-c2/2 * t_target) * jnp.cos(wd2 * t_target + 0.7*jnp.pi)
T_target2 = 2*jnp.pi/wd2

targets = jnp.stack([target1, target2])


# =====================================================
# Robot simulation before optimization
# =====================================================

# Initialize robot
N = 2 # number of segments
L = jnp.array([1.0e-1, 1.0e-1])
D = 1e0 * jnp.diag(jnp.array([1.0e-4, 1.0e-1, 1.0e-1,
                              1.0e-4, 1.0e-1, 1.0e-1]))
E = jnp.array([2e3, 2e3])
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L,
    "r": jnp.array([2e-2, 2e-2]),
    "rho": jnp.array([1070, 1070]),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": E,
    "G": 1e3 * jnp.ones((N,)),
    "D": D
}

robot = PlanarPCS_simple(
    num_segments = N,
    params = parameters,
    order_gauss = 5
)

# Simulation parameters
eps = jnp.finfo(jnp.float64).eps # machine epsilon
q0 = jnp.array([5.0*jnp.pi, 0.2, 0.1,
                5.0*jnp.pi, 0.2, 0.1]) # k, S_x, S_y
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = t0
t1 = t1
dt = 1e-4
save_at = t_target
solver = Tsit5() # Tsit5(), Euler(), Heun(), Midpoint(), Ralston(), Bosh3(), Dopri5(), Dopri8()
#step_size = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-3, force_dtmin=True) # ConstantStepSize(), PIDController(rtol=, atol=)
step_size = ConstantStepSize()
max_steps = int(1e5)
initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

# Simulate robot
print('Simulating robot...')
start = time.perf_counter()
sim_out = robot.rollout_to(
    initial_state= initial_state,
    u = u, 
    t1 = t1, 
    solver_dt = dt, 
    save_ts = save_at,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)
ts = sim_out.t
q_ts, _ = jnp.split(sim_out.y, 2, axis=1)
end = time.perf_counter()
print(f'Elapsed time (simulation, {solver}): {end-start} s')

# Compute chi = [th, x, y] for the two points in the GLOBAL frame. Shape (n_steps, 3)
chiG_1_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts, L[0])
chiG_2_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L))

# Compute the corresponding SE(2) matrix representations in the GLOBAL frame. Shape (n_steps, 3, 3)
TG_1_ts = jax.vmap(exp_SE2)(chiG_1_ts)
TG_2_ts = jax.vmap(exp_SE2)(chiG_2_ts)

# Compute SE(2) matrices in the LOCAL frames. Shape (n_steps, 3, 3)
@jax.jit
def global2local(T_global, T_ref):
    T_local = jnp.linalg.inv(T_ref) @ T_global
    return T_local

TL_1_ts = TG_1_ts # for first segment, local=global
TL_2_ts = jax.vmap(global2local, in_axes=(0,0))(TG_2_ts, TG_1_ts)

# Extract local x and y coordinates
xL_1_ts = TL_1_ts[:, 0, -1] # x coordinate for point 1
yL_2_ts = TL_2_ts[:, 1, -1] # y coordinate for point 2

# Compute index for discharging initial transient
idx_end_trans = onp.argmax(ts>2.0)

# Error
MSE1 = onp.mean((xL_1_ts[idx_end_trans:] - target1[idx_end_trans:]) ** 2)
MSE2 = onp.mean((yL_2_ts[idx_end_trans:] - target2[idx_end_trans:]) ** 2)
print(f'MSEs (before optimization): target 1 = {1e4*MSE1:.4f} cm^2 | target 2 = {1e4*MSE2:.4f} cm^2')

# Plot results
fig, ax = plt.subplots(2, 1)

ax[0].plot(ts, xL_1_ts, color='b', label=r'$x_{1}^{locA}(t)$')
ax[0].plot(t_target, target1, color='b', linestyle='--', label='target')
ax[0].set_xlabel('t [s]')
ax[0].set_ylabel(r'$x^{locA}$ [m]')
ax[0].grid(True)
ax[0].set_title('Local displacement point 1 (before optimization)')
ax[0].legend()

ax[1].plot(ts, yL_2_ts, color='b', label=r'$y_{2}^{locB}(t)$')
ax[1].plot(t_target, target2, color='b', linestyle='--', label='target')
ax[1].set_xlabel('t [s]')
ax[1].set_ylabel(r'$y^{locB}$ [m]')
ax[1].grid(True)
ax[1].set_title('Local displacement point 2 (before optimization)')
ax[1].legend()

fig.text(0.5, -0.05, f"L={L} m\n D={onp.diag(D)} Pa*s", ha="center", va="top")
fig.tight_layout()
fig.savefig(plots_folder/'Relative displacements (before optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = ts,
    q_list = q_ts,
    slider = True,
    animation = False,
    show = True
)


# =====================================================
# First optimization: match the periods
# =====================================================

# Loss function
def Loss(params_softplus):
    L = jax.nn.softplus(params_softplus[:N]) # get real, positive parameters
    #E = jax.nn.softplus(params_softplus[N:]) # get real, positive parameters
    #robot_updated = robot.update_params({"L": L, "E": E}) # update robot
    robot_updated = robot.update_params({"L": L}) # update robot

    # simulation
    sim_out = robot_updated.rollout_to(
        initial_state=initial_state,
        u = u, 
        t1 = t1, 
        solver_dt = dt, 
        save_ts = save_at,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    q_ts, _ = jnp.split(sim_out.y, 2, axis=1)

    # compute displacements in the local reference frames
    chiG_1_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, L[0])       # chi = [th, x, y] in the GLOBAL frame. Shape (n_steps, 3)
    chiG_2_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y] in the GLOBAL frame. Shape (n_steps, 3)

    TG_1_ts = jax.vmap(exp_SE2)(chiG_1_ts) # SE(2) matrix representation in the GLOBAL frame. Shape (n_steps, 3, 3)
    TG_2_ts = jax.vmap(exp_SE2)(chiG_2_ts) # SE(2) matrix representation in the GLOBAL frame. Shape (n_steps, 3, 3)

    TL_1_ts = TG_1_ts # SE(2) matrix in the LOCAL frame. Shape (n_steps, 3, 3). For first segment, local=global
    TL_2_ts = jax.vmap(global2local, in_axes=(0,0))(TG_2_ts, TG_1_ts) # SE(2) matrix in the LOCAL frame. Shape (n_steps, 3, 3)

    xL_1_ts = TL_1_ts[:, 0, -1] # extract local x coordinate for point 1
    yL_2_ts = TL_2_ts[:, 1, -1] # extract local y coordinate for point 2

    # # discharge initial transient
    xL_1_ts_short = xL_1_ts[idx_end_trans:]
    yL_2_ts_short = yL_2_ts[idx_end_trans:]

    # period estimation
    x = xL_1_ts_short - jnp.mean(xL_1_ts_short) # remove DC, so no peak at f=0 Hz
    y = yL_2_ts_short - jnp.mean(yL_2_ts_short) # remove DC, so no peak at f=0 Hz

    win = jnp.hanning(len(x)) # apply Hanning window, to reduce leakage
    x_w = x * win
    y_w = y * win

    N_pad = 4*len(x)
    X = jnp.fft.rfft(x_w, n=N_pad)    # FFT. Zero padding to increase frequency resolution
    Y = jnp.fft.rfft(y_w, n=N_pad)    # FFT. Zero padding to increase frequency resolution
    f = jnp.fft.rfftfreq(N_pad, d=dt) # new frequecy axis

    eps = jnp.finfo(jnp.float64).eps  # machine epsilon
    G_xx = jnp.abs(X)**2 + eps # compute power spectrum. Add eps for numerical stability
    G_yy = jnp.abs(Y)**2 + eps # compute power spectrum. Add eps for numerical stability
    G_xx = jnp.log(G_xx)       # use log to help softmax
    G_yy = jnp.log(G_yy)       # use log to help softmax

    f_nyq = 0.5 / dt 
    filter = f<=f_nyq
    G_xx = jnp.where(filter, G_xx, -1e12)
    G_yy = jnp.where(filter, G_yy, -1e12)

    weights1 = jax.nn.softmax(G_xx) # find peaks on the power spectrum
    weights2 = jax.nn.softmax(G_yy) # find peaks on the power spectrum

    f1_hat = jnp.sum(weights1*f) # estimate frequency basing on weighted sum (can't use argmax because of differentiability)
    f2_hat = jnp.sum(weights2*f) # estimate frequency basing on weighted sum (can't use argmax because of differentiability)
    T1_hat = 1.0 / f1_hat
    T2_hat = 1.0 / f2_hat

    J = jnp.mean((T1_hat - T_target1) ** 2) + jnp.mean((T2_hat - T_target2) ** 2)

    return J, (xL_1_ts, yL_2_ts, T1_hat, T2_hat)

#params = jnp.concatenate([robot.L, robot.E])
params = robot.L
params_softplus = InverseSoftplus(params) # inverse softplus

############################################################################
##### SOME CHECKS ##########################################################

# !! Plot loss shape !!
L1_range_softplus = (InverseSoftplus(0.1e-1).item(), InverseSoftplus(2.5e-1).item())
L2_range_softplus = (InverseSoftplus(0.05e-1).item(), InverseSoftplus(2.0e-1).item())
param_nan = PlotLoss(
    Loss=Loss,
    param1_range=L1_range_softplus,
    param2_range=L2_range_softplus,
    N1=15,         # 130
    N2=15,         # 128
    chunk_size=832, # 832
)
exit()
# !! End check !!


# # !! Check gradients computation !!
# import time
# loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))

# start = time.perf_counter()
# (loss, (x, y, T1_hat, T2_hat)), grads = loss_and_grad(params_softplus) # warmup
# print(loss, grads, x.shape, y.shape, T1_hat, T2_hat)                   # warmup
# end = time.perf_counter()
# print(f'time (warmup): {end-start} s')

# start = time.perf_counter()
# (loss, (x, y, T1_hat, T2_hat)), grads = loss_and_grad(params_softplus)
# print(loss, grads, x.shape, y.shape, T1_hat, T2_hat)
# end = time.perf_counter()
# print(f'time (already compiled): {end-start} s')
# exit()
# # !! End check !!

############################################################################
############################################################################

# Setup optimizer
lr_schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=25*2)
optimizer = optax.chain(
    optax.clip_by_global_norm(5e1),      # apply clipping
    optax.sgd(learning_rate=lr_schedule) # use SGD
)
opt_state = optimizer.init(params_softplus) # initialize optimizer

# Optimization iterations
print('Starting first optimization...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True)) # jit loss
(loss, others), grads = loss_and_grad(params_softplus)          # warm-up
n_iter = 25*2
solutions1_ts = onp.zeros((n_iter, len(target1)))
solutions2_ts = onp.zeros_like(solutions1_ts)
loss_ts = []
for i in tqdm(range(n_iter), 'First optimization'):
    params_print = jax.nn.softplus(params_softplus)
    (loss, (xL_1_ts, yL_2_ts, T1_hat, T2_hat)), grads = loss_and_grad(params_softplus)
    if jnp.isnan(loss):
        print('NaN loss')
        n_iter = i
        break
    
    #grads = grads.at[N:].multiply(1e6) # scale gradients of E to account for different order of magnitude
    updates, opt_state = optimizer.update(grads, opt_state, params_softplus)
    params_softplus = optax.apply_updates(params_softplus, updates)

    loss_ts.append(loss)
    solutions1_ts[i] = xL_1_ts
    solutions2_ts[i] = yL_2_ts
    tqdm.write(f"Iter {i:02d} | "
            f"loss={loss:.3e} | "
            f"L={params_print[:N]} | "
            #f"E={params_print[N:]} | "
            f"grads (before clipping)={grads} | "
            f"T_hat=[{T1_hat:.3f} ({T_target1:.3f}), {T2_hat:.3f} ({T_target2:.3f})]")

L_opt = jax.nn.softplus(params_softplus[:N])
#E_opt = jax.nn.softplus(params_softplus[N:])
#print(f"Optimal L: {L_opt} | Optimal E: {E_opt}")
print(f"Optimal L: {L_opt}")

# Save optimal parameters
#onp.savez(data_folder/'after_1_optimiz', L=onp.array(L_opt), E=onp.array(E_opt))
onp.savez(data_folder/'after_1_optimiz', L=onp.array(L_opt))

# Visualization
plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(plots_folder/'Loss1')
#plt.show()

plot_optimiz_evolution(
    ts=ts,
    solutions=onp.stack((solutions1_ts, solutions2_ts), axis=0),
    targets=targets,
    show=True,
    savepath=plots_folder/'optimization1_animation.gif'
)
exit()


# =====================================================
# Second optimization: match the entire signals
# =====================================================

# Load saved data from previous optimization and update robot
data_1 = onp.load(data_folder/'after_1_optimiz.npz')
L = jnp.array(data_1['L'])
robot = robot.update_params({"L": L})

# Loss function
def Loss(params_softplus):
    params = jax.nn.softplus(params_softplus)     # get real, positive parameters
    D = jnp.diag(params)
    robot_updated = robot.update_params({"D": D}) # update robot

    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        save_dt = save_dt, 
        solver = solver,
        max_steps = max_steps
    )
    
    # compute displacements in the local reference frames
    chiG_1_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, L[0])       # chi = [th, x, y] in the GLOBAL frame. Shape (n_steps, 3)
    chiG_2_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y] in the GLOBAL frame. Shape (n_steps, 3)

    TG_1_ts = jax.vmap(exp_SE2)(chiG_1_ts) # SE(2) matrix representation in the GLOBAL frame. Shape (n_steps, 3, 3)
    TG_2_ts = jax.vmap(exp_SE2)(chiG_2_ts) # SE(2) matrix representation in the GLOBAL frame. Shape (n_steps, 3, 3)

    TL_1_ts = TG_1_ts # SE(2) matrix in the LOCAL frame. Shape (n_steps, 3, 3). For first segment, local=global
    TL_2_ts = jax.vmap(global2local, in_axes=(0,0))(TG_2_ts, TG_1_ts) # SE(2) matrix in the LOCAL frame. Shape (n_steps, 3, 3)

    xL_1_ts = TL_1_ts[:, 0, -1] # extract local x coordinate for point 1
    yL_2_ts = TL_2_ts[:, 1, -1] # extract local y coordinate for point 2

    # discharge initial transient
    xL_1_ts_short = xL_1_ts[idx_end_trans:]
    yL_2_ts_short = yL_2_ts[idx_end_trans:]

    J = 1e3 * jnp.mean((xL_1_ts_short - target1[idx_end_trans:]) ** 2) + 1e3 * jnp.mean((yL_2_ts_short - target2[idx_end_trans:]) ** 2)

    return J, (xL_1_ts, yL_2_ts)

params = jnp.diag(D)                      # initial parameters
params_softplus = InverseSoftplus(params) # inverse softplus

# # !! Check gradients computation !!
# loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))
# (loss, (x, y)), grads = loss_and_grad(params_softplus)
# print(loss, grads, x.shape, y.shape)
# exit()

# Setup optimizer
# lr_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=0.0,
#     peak_value=1e-2,
#     warmup_steps=15,
#     decay_steps=75-15
# )
lr_schedule = optax.piecewise_constant_schedule(init_value=1e-1, boundaries_and_scales={25: 0.5, 50: 0.5})
optimizer = optax.sgd(learning_rate=lr_schedule) # use SGD
opt_state = optimizer.init(params_softplus)      # initialize optimizer

# Optimization iterations
print('Starting second optimization...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True)) # jit loss
(loss, others), grads = loss_and_grad(params_softplus)          # warm-up
n_iter = 5
loss_ts = []
solutions1_ts = onp.zeros((n_iter, len(target1)))
solutions2_ts = onp.zeros((n_iter, len(target2)))
for i in tqdm(range(n_iter), 'Second optimization'):
    params_print = jax.nn.softplus(params_softplus)
    (loss, (xL_1_ts, yL_2_ts)), grads = loss_and_grad(params_softplus)
    if jnp.isnan(loss):
        print('NaN loss')
        n_iter = i
        break
    
    updates, opt_state = optimizer.update(grads, opt_state, params_softplus)
    params_softplus = optax.apply_updates(params_softplus, updates)

    loss_ts.append(loss)
    solutions1_ts[i] = xL_1_ts
    solutions2_ts[i] = yL_2_ts
    if i % 1 == 0:
        print(
            f"Iter {i:02d} | "
            f"loss={loss:.3e} | "
            f"D={params_print.flatten()} Pa*s | "
            f"grads={grads.squeeze()}"
        )

D_opt = jnp.diag(jax.nn.softplus(params_softplus))
print("Optimal D:\n", D_opt)

# Save optimal parameters
onp.savez(data_folder/'after_2_optimiz', D=onp.array(D_opt))

# Visualization
plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(plots_folder/'Loss2')
#plt.show()

plot_optimiz_evolution(
    ts=ts,
    solutions1_ts=solutions1_ts,
    solutions2_ts=solutions2_ts,
    targets=targets,
    show=True,
    save=True,
    savepath=plots_folder/'optimization2_animation.gif'
)


# =====================================================
# Robot simulation after optimization
# =====================================================

# Update robot with optimal parameters
data_1 = onp.load(data_folder/'after_1_optimiz.npz')
data_2 = onp.load(data_folder/'after_2_optimiz.npz')
L_opt = jnp.array(data_1['L'])
D_opt = jnp.array(data_2['D'])
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

# Simulate the optimized robot
ts, q_ts, _ = robot_opt.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    save_dt = save_dt, 
    solver = solver,
    max_steps = max_steps
)

# Compute chi = [th, x, y] for the two points in the GLOBAL frame. Shape (n_steps, 3)
chiG_1_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts, L_opt[0])
chiG_2_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L_opt))

# Compute the corresponding SE(2) matrix representations in the GLOBAL frame. Shape (n_steps, 3, 3)
TG_1_ts = jax.vmap(exp_SE2)(chiG_1_ts)
TG_2_ts = jax.vmap(exp_SE2)(chiG_2_ts)

# Compute SE(2) matrices in the LOCAL frames. Shape (n_steps, 3, 3)
TL_1_ts = TG_1_ts # for first segment, local=global
TL_2_ts = jax.vmap(global2local, in_axes=(0,0))(TG_2_ts, TG_1_ts)

# Extract local x and y coordinates
xL_1_ts = TL_1_ts[:, 0, -1] # x coordinate for point 1
yL_2_ts = TL_2_ts[:, 1, -1] # y coordinate for point 2

# Error
MSE1 = onp.mean((xL_1_ts[idx_end_trans:] - target1[idx_end_trans:]) ** 2)
MSE2 = onp.mean((yL_2_ts[idx_end_trans:] - target2[idx_end_trans:]) ** 2)
print(f'MSEs (after optimization): target 1 = {1e4*MSE1:.4f} cm^2 | target 2 = {1e4*MSE2:.4f} cm^2')

# Plot results
fig, ax = plt.subplots(2, 1)

ax[0].plot(ts, xL_1_ts, color='b', label=r'$x_{1}^{locA}(t)$')
ax[0].plot(t_target, target1, color='b', linestyle='--', label='target')
ax[0].set_xlabel('t [s]')
ax[0].set_ylabel(r'$x^{locA}$ [m]')
ax[0].grid(True)
ax[0].set_title('Local displacement point 1 (after optimization)')
ax[0].legend()

ax[1].plot(ts, yL_2_ts, color='b', label=r'$y_{2}^{locB}(t)$')
ax[1].plot(t_target, target2, color='b', linestyle='--', label='target')
ax[1].set_xlabel('t [s]')
ax[1].set_ylabel(r'$y^{locB}$ [m]')
ax[1].grid(True)
ax[1].set_title('Local displacement point 2 (after optimization)')
ax[1].legend()

fig.text(0.5, -0.05, f"L={L_opt} m\n D={onp.diag(D_opt)} Pa*s", ha="center", va="top")
fig.tight_layout()
fig.savefig(plots_folder/'Relative displacements (after optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot_opt,
    t_list = ts,
    q_list = q_ts,
    slider = True,
    animation = False,
    show = True
)
