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
from soromox.systems.pcs import PCS
from my_utilis import PlotLoss

jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

# Functions for plotting
def draw_robot_curve(
    batched_forward_kinematics,
    L_max: float,
    q,
    num_points: int = 50,
):
    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)

def animate_robot_matplotlib(
    robot: PCS,
    t_list,  # shape (T,)
    q_list,  # shape (T, DOF)
    target = None,
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError("Cannot use both animation and slider at the same time. Choose one.")
    
    width = jnp.linalg.norm(robot.L) * 3
    height = width
    
    if target is not None:
        t_old = onp.linspace(0, 1, len(target))
        t_new = onp.linspace(0, 1, len(q_list))
        target = onp.interp(t_new, t_old, target)

        x_plane = onp.array([[-width/2, width/2], [-width/2, width/2]])
        z_plane = onp.array([[0,0],[height,height]])

    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = jnp.sum(robot.L)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if animation:
        (line,) = ax.plot([], [], [], lw=4, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return line, target, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve = draw_robot_curve(batched_forward_kinematics, L_max, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            line.set_3d_properties(curve[:, 2])
            if target is not None:
                y_target = target[frame_idx]
                y_plane = jnp.full_like(x_plane, y_target)
                if hasattr(update, "plane"):
                    update.plane.remove()
                update.plane = ax.plot_surface(x_plane, y_plane, z_plane, color='r', alpha=0.3)
            title_text.set_text(f"t = {t:.2f} s")
            return (line, title_text) + ((update.plane,) if target is not None else ())

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
            ax.cla()  # clear current axes
            if target is not None:
                y_target = target[frame_idx]
                y_plane = jnp.full_like(x_plane, y_target)
                ax.plot_surface(x_plane, y_plane, z_plane, color='r', alpha=0.3, label='target plane')
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(-width / 2, width / 2)
            ax.set_zlim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            q = q_list[frame_idx]
            curve = draw_robot_curve(batched_forward_kinematics, L_max, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
            fig.canvas.draw_idle()

        # create slider
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

        update_plot(0)  # initial plot

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
    # create slider animation   
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)

    ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.03])  # [left, bottom, width, height]

    def update_plot(frame_idx):
        ax.cla()  # clear current axes
        ax.plot(ts, solutions_ts[frame_idx, :], color='r', label='simulation')
        ax.plot(ts, target, color='r', linestyle='--', label='target')
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(onp.min([solutions_ts.min(), target.min()]), onp.max([solutions_ts.max(), target.max()]))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("$y [m]")
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
    
    # create animation to be saved
    if save and savepath is None:
        print('Optimization animation not saved: must provide a path where the animation can be saved.')

    elif save and savepath is not None:        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        (line_sim,) = ax.plot([], [], color="r", label='simulation')
        (line_tar,) = ax.plot([], [], color="r", linestyle='--', label='target')
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(onp.min([solutions_ts.min(), target.min()]), onp.max([solutions_ts.max(), target.max()]))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("y [m]")
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


# =====================================================
# Target creation
# =====================================================

# End effector target (y coordinate)
t0 = 0.0
t1 = 5.0
dt_target = 1e-3
t_target = jnp.arange(t0, t1, dt_target)
c = 0.5
k = 30
wd = jnp.sqrt(4*k-c**2)/2
target = 0.1 * jnp.exp(-c/2 * t_target) * jnp.cos(wd * t_target)


# =====================================================
# Robot simulation before optimization
# =====================================================

# Initialize robot
N = 1                                                         # number of segments
L = 1e-1 * jnp.ones((N,))                                     # segments length
D = jnp.diag(jnp.array([1e-4, 1e-4, 1e-4, 1e-1, 1e-1, 1e-1])) # damping matrix
parameters = {
    "p0": jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0, 0.0, 0.0]),
    "L": L,
    "r": 2e-2 * jnp.ones((N,)),
    "rho": 1070 * jnp.ones((N,)),
    "g": jnp.array([0.0, 0.0, 9.81]),
    "E": 2e3 * jnp.ones((N,)),
    "G": 1e3 * jnp.ones((N,)),
    "D": D
}

robot = PCS(
    num_segments = N,
    params = parameters,
    order_gauss = 5
)

# Simulation parameters
q0 = jnp.tile(jnp.array([0.0, 5.0*jnp.pi, 0.0, 0.1, 0.2, 0.0]), N)
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = t0
t1 = t1
dt0 = 5e-4
solver = Tsit5() # Tsit5(), Euler(), Heun(), Midpoint(), Ralston(), Bosh3(), Dopri5(), Dopri8()
#step_size = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-3, force_dtmin=True) # ConstantStepSize(), PIDController(rtol=, atol=)
step_size = ConstantStepSize()
max_steps = int(1e5)

# Simulate robot
start = time.perf_counter()
_, q_ts, _ = robot.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt0, 
    saveat_ts = t_target,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]
end = time.perf_counter()
print(f'Elapsed time (simulation, {solver}): {end-start} s')

# Compute MSE
MSE = onp.mean((y_ts - target) ** 2)
print(f'MSE before optimization: {1e4*MSE:.4f} cm^2')

# Plot results
plt.figure()
plt.plot(t_target, ee_ts[:,0], color='b', label='x end effector')
plt.plot(t_target, ee_ts[:,1], color='r', label=f'y end effector')
plt.plot(t_target, ee_ts[:,2], color='g', label='z end effector')
plt.plot(t_target, target, color='r', linestyle='--', label=f'target y')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.grid(True)
plt.title(f'End effector position (before optimization)')
plt.figtext(0.5, -0.05, f"L={L[0]:.3f} m\n D={onp.diag(D)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (before optimization)')
#plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = t_target,
    q_list = q_ts,
    target = target,
    slider = True,
    animation = False,
    show = True
)
exit()


# =====================================================
# Optimization
# =====================================================

# Loss function
def Loss(params):
    # extract optimization parameters and update robot
    L = jnp.array([params[0]])     
    #D = jnp.diag(params[1:])           
    D = params[1] * jnp.diag(jnp.array([1e-4, 1e-4, 1e-4, 1e-1, 1e-1, 1e-1]))
    robot_updated = robot.update_params({"L": L, "D": D})

    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt0, 
        saveat_ts = t_target, 
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    g_ee_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L)) # time history of the SE(3) matrices. Shape (n_steps, 4, 4)
    ee_ts = g_ee_ts[:,:3,-1]                                                                # time history of position [x,y,z]. Shape (n_steps, 3)
    y_ts = ee_ts[:,1]                                                                       # time history of y coordinate. Shape (n_steps,)

    # MSE wrt the target
    J = jnp.mean((y_ts - target) ** 2)

    return J, (y_ts)

# First guess for the parameters
#params = jnp.concatenate([L, jnp.diag(D).flatten()])
params = jnp.array([L[0], 1e0])

############################################################################
##### SOME CHECKS ##########################################################

# !! Plot loss shape !!
L_range = (0.1e-1, 2e-1)
D_range = (1e-1, 1e1)
param_nan = PlotLoss(
    Loss=Loss,
    param1_range=L_range,
    param2_range=D_range,
    N1=10,         # 130
    N2=10,         # 128
    chunk_size=832, # 832
)
exit()
# !! End check !!


# # !! Check gradients computation !!
# import time
# loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))

# start = time.perf_counter()
# (loss, (y_ts)), grads = loss_and_grad(params) # warm-up
# print(loss, grads, y_ts.shape)                # warm-up
# end = time.perf_counter()
# print(f'time (warmup): {end-start} s')

# start = time.perf_counter()
# (loss, (y_ts)), grads = loss_and_grad(params)
# print(loss, grads, y_ts.shape)
# end = time.perf_counter()
# print(f'time (already compiled): {end-start} s')

# exit()
# # !! End check !!

############################################################################
############################################################################

# Setup optimizer
# lr_schedule = optax.piecewise_constant_schedule(init_value=1e-1, boundaries_and_scales={45: 0.1, 90: 0.1})
# optimizer = optax.adam(learning_rate=lr_schedule)  # use ADAM
optimizer = optax.sgd(learning_rate=1e-3) # use SGD
opt_state = optimizer.init(params)        # initialize optimizer

# Optimization iterations
print('Warm-up loss function...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))
_, _ = loss_and_grad(params) # warm-up

n_iter = 10
loss_ts = []
solutions_ts = onp.zeros((n_iter, len(target)))
for i in tqdm(range(n_iter), 'Optimization'):
    params_print = params
    (loss, (y_ts)), grads = loss_and_grad(params)
    if jnp.isnan(loss):
        print('NaN loss')
        n_iter = i
        break
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    loss_ts.append(loss)
    solutions_ts[i] = y_ts
    tqdm.write(f"Iter {i:02d} | " 
               f"L={params[0]} | "
               f"D={params[1:]} | "
               f"loss={loss:.3e} | "
               f"grads={grads}")

L_opt = jnp.array([params[0]])
D_opt = jnp.diag(params[1:])
print(f"Optimal L: {L_opt} m | Optimal D: {D_opt.flatten()} Pa*s")

# Visualization
plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(plots_folder/'Loss', bbox_inches='tight')
plt.show()

plot_optimiz_evolution(
    ts=t_target,
    solutions_ts=solutions_ts,
    target=target,
    show=True,
    save=True,
    savepath=plots_folder/'optimization_animation.gif'
)
exit()


# =====================================================
# Robot simulation after optimization
# =====================================================

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

# Simulate the optimized robot
ts, q_ts, qd_ts = robot_opt.resolve_upon_time(
    q0=q0,
    qd0=qd0,
    u=u,
    t0=t0,
    t1=t1,
    dt=dt0,
    saveat_ts=t_target,
    solver=solver,
    stepsize_controller=step_size,    
    max_steps=None,
)
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label='y end effector')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.plot(t_target, target, color='r', linestyle='--', label=f'target y')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.grid(True)
plt.title(f'End effector position (after optimization)')
plt.figtext(0.5, -0.05, f"L={L_opt.item():.3f} m\n D={onp.diag(D_opt)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (after optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot_opt,
    t_list = ts,
    q_list = q_ts,
    target = target,
    slider = True,
    animation = False,
    show = True
)