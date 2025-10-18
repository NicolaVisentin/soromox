# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports and setup
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
import pickle

from pathlib import Path
from tqdm import tqdm
import time
import sys
import select

from soromox.systems.planar_pcs import PlanarPCS
from soromox.systems.planar_pcs_simplified import PlanarPCS_simple
from soromox.utils.lie_algebra.se2 import exp_SE2
from my_utilis import PlotLoss, InverseSoftplus

jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

# Functions for plotting
def draw_robot(robot: PlanarPCS | PlanarPCS_simple, q: Array, num_points: int = 50):
    L_max = jnp.sum(robot.L)
    s_ps = jnp.linspace(0, L_max, num_points)

    chi_ps = robot.forward_kinematics_batched(q, s_ps)  # (N,3)
    curve = onp.array(chi_ps[:, 1:], dtype=onp.float64) # (N,2)
    pos_tip = curve[-1]                                 # [x_tip, y_tip]

    return curve, pos_tip

def animate_robot_matplotlib(
    robot: PlanarPCS | PlanarPCS_simple,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    target: Array = None,
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

    width = jnp.linalg.norm(robot.L) * 3
    height = width

    if target is not None:
        t_old = onp.linspace(0, 1, len(target))
        t_new = onp.linspace(0, 1, len(q_list))
        target = onp.interp(t_new, t_old, target)

    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_base(ax, robot, L=0.1)

    if animation:
        (line,) = ax.plot([], [], lw=4, color="blue")
        (tip,) = ax.plot([], [], 'ro', markersize=5)
        (targ,) = ax.plot([], [], color='r', alpha=0.5)
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        ax.grid(True)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            tip.set_data([], [])
            targ.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, tip, targ, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            tip.set_data([tip_pos[0]], [tip_pos[1]])
            if target is not None:
                x_target = target[frame_idx]
                targ.set_data([x_target,x_target], [0,height])
            title_text.set_text(f"t = {t:.2f} s")
            return (line, tip, title_text) + ((targ,) if target is not None else ())

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
            q = q_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            ax.plot([tip_pos[0]], [tip_pos[1]], 'ro', markersize=5)
            if target is not None:
                x_target = target[frame_idx]
                ax.plot([x_target,x_target], [0,height], 'r', alpha=0.5)
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

def plot_optimiz_evolution(ts, solutions_ts, target, show=True, save=False, savepath=None):
    """
    Args:
        ts:
            Time vector. Shape (N_timeinstants)
        solutions_ts:
            Shape (N_iterations, N_timeinstants)
        target:
            Shape (N_timeistants,)
        show:
            Boolean.
        save:
            Boolean.
        savepath:
            Path where saving animation, in form: folder/'title_animation.gif'
    """
    # Limits for the y axis
    y_lower = onp.min([solutions_ts[onp.isfinite(solutions_ts) & (solutions_ts>-1e6)].min(), target.min()])
    y_upper = onp.max([solutions_ts[onp.isfinite(solutions_ts) & (solutions_ts<1e6)].max(), target.max()])

    # Create slider animation   
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)
    ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.03])  # [left, bottom, width, height]

    def update_plot(frame_idx):
        ax.cla()  # clear current axes
        ax.plot(ts, solutions_ts[frame_idx, :], color='r', label='simulation')
        ax.plot(ts, target, color='r', linestyle='--', label='target')
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"Optimization (iteration n. {frame_idx})")
        ax.grid(True)
        ax.legend()
        fig.canvas.draw_idle()

    slider = Slider(
        ax=ax_slider,
        label="Iter.",
        valmin=0,
        valmax=len(solutions_ts) - 1,
        valinit=0,
        valstep=1,
    )
    slider.on_changed(update_plot)

    update_plot(0)  # initial plot

    if show:
        plt.show()
    plt.close(fig)
    
    # Create animation to be saved
    if save and savepath is None:
        print('Optimization animation not saved: must provide a path where the animation can be saved.')

    elif save and savepath is not None:        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        (line_sim,) = ax.plot([], [], color="r", label='simulation')
        (line_tar,) = ax.plot([], [], color="r", linestyle='--', label='target')
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x [m]")
        title_text = ax.set_title(f"Optimization (iteration n. 0)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        def init():
            line_sim.set_data([], [])
            line_tar.set_data(ts, target)
            title_text.set_text(f"Optimization (iteration n. 0)")
            return line_sim, line_tar, title_text

        def update(frame_idx):
            line_sim.set_data(ts, solutions_ts[frame_idx, :])
            title_text.set_text(f"Optimization (iteration n. {frame_idx})")
            return (line_sim, line_tar, title_text)
        
        ani = FuncAnimation(
            fig,
            update,
            frames=len(solutions_ts),
            init_func=init,
            blit=True
        )

        ani.save(savepath, writer="pillow", fps=len(solutions_ts)/10)
        plt.close(fig)

# Folder for plots and videos
curr_folder = Path(__file__).parent
plots_folder = curr_folder/'plots and videos'/Path(__file__).stem
plots_folder.mkdir(parents=True, exist_ok=True)

# Folder for saving data
data_folder = curr_folder/'saved data'/Path(__file__).stem
data_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Target creation
# =====================================================

# End effector target (x coordinate)
t0 = 0.0
t1 = 5.0
dt_target = 1e-3
t_target = jnp.arange(t0, t1, dt_target)

c = 0.5
k = 29
wd = jnp.sqrt(4*k-c**2)/2
target = 0.2 * jnp.exp(-c/2 * t_target) * jnp.cos(wd * t_target)


# =====================================================
# Robot simulation before optimization
# =====================================================
print('--- INITIAL SIMULATION ---')

# Select planar PCS model (with or without Coriolis effect)
RobotModel = PlanarPCS_simple # PlanarPCS, PlanarPCS_simple

# Initialize robot
N = 2 # number of segments

L_default = jnp.array([1.0e-1, 1.0e-1])
L_scale = 1.6
L = L_scale * L_default

D_default = jnp.diag(jnp.array([1.0e-4, 1.0e-1, 1.0e-1,
                                1.0e-4, 1.0e-1, 1.0e-1]))
D_scale = 2
D = D_scale * D_default

parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L,
    "r": jnp.array([2e-2, 2e-2]),
    "rho": jnp.array([1070, 1070]),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": jnp.array([2e3, 2e3]),
    "G": 1e3 * jnp.ones((N,)),
    "D": D
}

robot = RobotModel(
    num_segments = N,
    params = parameters,
    order_gauss = 5
)

# Simulation parameters
eps = jnp.finfo(jnp.float64).eps # machine epsilon
q0 = jnp.array([-5.0*jnp.pi, 0.2, 0.1,
                5.0*jnp.pi, 0.2, 0.1]) # k, S_x, S_y
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = t0
t1 = t1
dt =1e-5
save_at = t_target
solver = Euler() # Tsit5(), Euler(), Heun(), Midpoint(), Ralston(), Bosh3(), Dopri5(), Dopri8()
#step_size = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-4, force_dtmin=True) # ConstantStepSize(), PIDController(rtol=, atol=)
step_size = ConstantStepSize()
max_steps = int(1e6)

# Simulate robot
print('Simulating robot...')
start = time.perf_counter()
ts, q_ts, _ = robot.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    saveat_ts = save_at,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)
end = time.perf_counter()
print(f'Elapsed time (simulation): {end-start} s')

# Extract end effector x coordinate in time
chi_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y]. Shape (n_steps, 3)
x_ee_ts = chi_ee_ts[:,1]

# Compute index for discharging initial transient
idx_end_trans = onp.argmax(ts>1.0)

# Error
MSE = onp.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2)
print(f'MSE before optimization: {1e4*MSE:.4f} cm^2')

# Plot results
plt.figure()
plt.plot(t_target, x_ee_ts, color='b', label='x end effector')
plt.plot(t_target, target, color='b', linestyle='--', label='target')
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.title(f'End effector position (before optimization)')
plt.figtext(0.5, -0.05, f"L={onp.array(L)} m\n D={onp.diag(D)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (before optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = t_target,
    q_list = q_ts,
    target = target,
    interval = 1e-2, 
    slider = True,
    animation = False,
    show = True
)


# =====================================================
# Optimization
# =====================================================

# First guess for the parameters. Parameters are scaling factors L_scale and D_scale for L and D.
# They are forced to be positive and > than a certain threshold L_scale_min and D_scale_min.
params = jnp.array([L_scale, D_scale])
L_scale_min = 1e-2
D_scale_min = 1e-4

params_min = jnp.array([L_scale_min, D_scale_min])
params_softplus = InverseSoftplus(params-params_min)

# Loss function
def Loss(params_softplus):
    # update robot
    params = params_min + jax.nn.softplus(params_softplus)
    L_scale = params[0]
    D_scale = params[1]
    L = L_default * L_scale
    D = D_default * D_scale
    robot_updated = robot.update_params({"L": L, "D": D})

    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        saveat_ts = save_at,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )

    # extract end effector x coordinate in time
    chi_ee_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y]. Shape (n_steps, 3)
    x_ee_ts = chi_ee_ts[:,1]

    J = jnp.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2) # ! discharge initial transient

    return J, (x_ee_ts)

############################################################################
##### SOME CHECKS ##########################################################

# # !! Check (jitted) loss computation !!
# loss_fn = jax.jit(Loss)

# start = time.perf_counter()
# loss, x = loss_fn(params_softplus) # warmup
# print(loss, x.shape)         # warmup
# end = time.perf_counter()
# print(f'time (warmup): {end-start} s')

# start = time.perf_counter()
# loss, x = loss_fn(params_softplus)
# print(loss, x.shape)
# end = time.perf_counter()
# print(f'time (already compiled): {end-start} s')
# exit()
# # !! End check !!

##########

# # !! Check loss + gradients computation !!
# loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))

# start = time.perf_counter()
# (loss, (x)), grads = loss_and_grad(params_softplus) # warmup
# print(loss, grads, x.shape)                         # warmup
# end = time.perf_counter()
# print(f'time (warmup): {end-start} s')

# start = time.perf_counter()
# (loss, (x)), grads = loss_and_grad(params_softplus)
# print(loss, grads, x.shape)
# end = time.perf_counter()
# print(f'time (already compiled): {end-start} s')
# exit()
# # !! End check !!

##########

# # !! Plot loss shape !!
# par1_range = jnp.array([0.6, 2.5])
# par2_range = jnp.array([0.8, 10])

# par1_range_softplus = (InverseSoftplus(par1_range[0] - L_scale_min).item(), InverseSoftplus(par1_range[1] - L_scale_min).item())
# par2_range_softplus = (InverseSoftplus(par2_range[0] - D_scale_min).item(), InverseSoftplus(par2_range[1] - D_scale_min).item())
# _, _, _ = PlotLoss(
#     Loss=Loss,
#     param1_range=par1_range_softplus,
#     param2_range=par2_range_softplus,
#     N1=76,         # 130
#     N2=76,         # 128
#     chunk_size=832, # 832
#     min_softplus=(L_scale_min, D_scale_min),
#     savepath=plots_folder/'Loss_surface'
# )
# exit()
# with open(plots_folder/'Loss_surface_(0.6-2.5)_(0.8-10)_76x76', 'rb') as f: # if you want to load and plot a saved surface:
#     fig = pickle.load(f) 
#     ax = fig.gca()
#     ax.set_xlim([1.25, 2])
#     ax.set_ylim([7, 10])
#     ax.set_zlim([0, 0.01])
#     plt.show()
# exit()
# # !! End check !!

############################################################################
############################################################################
print(F'\n--- OPTIMIZATION ---')

# Setup optimizer
lr_scheduling = optax.piecewise_constant_schedule(
    init_value=5e-1,
    boundaries_and_scales={10: 2, 20: 5}
)
optimizer = optax.sgd(learning_rate=lr_scheduling)
opt_state = optimizer.init(params_softplus) # initialize optimizer

# Warm-up loss function
print('Warm-up loss function...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True)) # jit loss
(loss, others), grads = loss_and_grad(params_softplus)          # warm-up

# Optimization iterations
n_iter = 25
solutions_ts = onp.empty((0, len(t_target)))
loss_ts = []
pbar = tqdm(range(n_iter), 'Optimization (enter "q" to stop)')
for ii in pbar:
    params_print = params_min + jax.nn.softplus(params_softplus) # current parameters
    (loss, (x_ee_ts)), grads = loss_and_grad(params_softplus)    # current loss and grads and solution 
    loss_ts.append(loss)                                         # save the loss
    solutions_ts = onp.vstack([solutions_ts, x_ee_ts])           # save the solution
    tqdm.write(
        f"Iter {ii:02d} | "
        f"L={params_print[0]*L_default} | "
        f"D={params_print[1]*onp.diag(D_default)} | "
        f"loss={loss:.3e} | "
        f"grads={grads}"
    )
    if jnp.isnan(loss): # if a NaN is detected don't update parameters and exit the for loop
        pbar.close()
        print('NaN loss')
        n_iter = ii+1
        break
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]: # if the user presses "q" don't update parameters and exit the for loop
        pbar.close()
        cmd = sys.stdin.readline().strip()
        if cmd.lower() == 'q':
            print('Interrupted by user')
            n_iter = ii+1
            break
    updates, opt_state = optimizer.update(grads, opt_state, params_softplus) # update params_softplus with the gradients
    params_softplus = optax.apply_updates(params_softplus, updates)

# Extract and save optimal parameters
params_opt = params_min + jax.nn.softplus(params_softplus)
L_opt = L_default * params_opt[0]
D_opt = D_default * params_opt[1]
print(f"Optimal L: {L_opt} | Optimal D: {onp.diag(D_opt)}")
onp.savez(data_folder/'after_optimiz', L=onp.array(L_opt), D=onp.array(D_opt))

# Visualization
plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(plots_folder/'Loss')
#plt.show()

plot_optimiz_evolution(
    ts=t_target,
    solutions_ts=solutions_ts,
    target=target,
    show=True,
    save=True,
    savepath=plots_folder/'optimization_animation.gif'
)


# =====================================================
# Robot simulation after optimization
# =====================================================
print('\n--- FINAL SIMULATION ---')

# Update robot with optimal parameters
data_opt = onp.load(data_folder/'after_optimiz.npz')
L_opt = jnp.array(data_opt['L'])
D_opt = jnp.array(data_opt['D'])
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

# Simulate robot
print('Simulating robot...')
_, q_ts, _ = robot_opt.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    saveat_ts = save_at,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)

# Extract end effector x coordinate in time
chi_ee_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L_opt)) # chi = [th, x, y]. Shape (n_steps, 3)
x_ee_ts = chi_ee_ts[:,1]

# Error
MSE = onp.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2)
print(f'MSE after optimization: {1e4*MSE:.4f} cm^2')

# Plot results
plt.figure()
plt.plot(t_target, x_ee_ts, color='b', label='x end effector')
plt.plot(t_target, target, color='b', linestyle='--', label='target')
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.title(f'End effector position (after optimization)')
plt.figtext(0.5, -0.05, f"L={L_opt} m\n D={onp.diag(D_opt)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (after optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot_opt,
    t_list = t_target,
    q_list = q_ts,
    target = target,
    interval = 1e-2, 
    slider = True,
    animation = False,
    show = True
)