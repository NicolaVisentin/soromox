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
from diffrax import Tsit5
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from soromox.systems.pcs import PCS
from my_utilis import InverseSoftplus

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
    if slider:
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.03])  # [left, bottom, width, height]

    def update_plot(frame_idx):
        ax.cla()  # clear current axes
        ax.plot(ts, solutions_ts[frame_idx, :], color='r', label='simulation')
        ax.plot(ts, target, color='r', linestyle='--', label='target')
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(onp.min([solutions_ts.min(), target.min()]), onp.max([solutions_ts.max(), target.max()]))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("y [m]")
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
t1 = 10.0
dt = 1e-3
t_target = jnp.arange(t0, t1, dt)
c = 0.5
k = 10
wd = jnp.sqrt(4*k-c**2)/2
target = 0.75 * jnp.exp(-c/2 * t_target) * jnp.cos(wd * t_target)


# =====================================================
# Robot simulation before optimization
# =====================================================

# Initialize robot
N = 1                     # segments
L = 3e-1 * jnp.ones((N,)) # segments length
D = 1e-3 * jnp.diag((jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), N, axis=0) * L[:, None]).flatten())
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
q0 = jnp.tile(jnp.array([0.0, 3.0, 0.0, 1.0, 0.0, 0.0]), N) # ! nan gradients if q[0]=q[1]=q[2]=0.0
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = t0
t1 = t1
dt = dt
skip_step = 1
solver = Tsit5()
max_steps = int(1e5)

# Simulate robot
ts, q_ts, _ = robot.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    skip_steps = skip_step, 
    solver = solver,
    max_steps = max_steps
)
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]

# Error
MSE = onp.mean((y_ts - target) ** 2)
print(f'MSE (before optimization) = {MSE:.4f} m^2')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label=f'y end effector')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.plot(t_target, target, color='r', linestyle='--', label=f'target y')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.grid(True)
plt.title(f'End effector position (before optimization)')
plt.figtext(0.5, -0.05, f"L={L[0]:.3f} m\n D={onp.diag(D)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (before optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = ts,   # shape (T,)
    q_list = q_ts, # shape (T, DOF)
    target = target,
    slider = True,
    animation = False,
    show = True
)


# =====================================================
# Optimization
# =====================================================

# Loss function
def Loss(params_softplus):
    params = jax.nn.softplus(params_softplus) # get real, positive L
    L = jnp.array([params[0]])                # extract optimization parameters
    D = jnp.diag(params[1:])
    robot_updated = robot.update_params({"L": L, "D": D}) # update robot

    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        skip_steps = skip_step, 
        solver = solver,
        max_steps = max_steps
    )
    g_ee_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L))
    ee_ts = g_ee_ts[:,:3,-1]
    y_ts = ee_ts[:,1]

    J = 1e2 * jnp.mean((y_ts - target) ** 2)

    return J, y_ts

params = jnp.concatenate([L, jnp.diag(D).flatten()])    # initial parameters
params_softplus = InverseSoftplus(params) # inverse softplus

# # !! Check gradients computation !!
# loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))
# (loss, y_ts), grads = loss_and_grad(params_softplus)
# print(loss, grads, y_ts.shape)
# exit()

# Setup optimizer
lr_schedule = optax.piecewise_constant_schedule(init_value=1e-1, boundaries_and_scales={45: 0.1, 90: 0.1})
optimizer = optax.adam(learning_rate=lr_schedule)  # use ADAM
opt_state = optimizer.init(params_softplus)        # initialize optimizer

# Optimization iterations
print('Starting optimization...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss, has_aux=True))
n_iter = 15
loss_ts = []
solutions_ts = onp.zeros((n_iter, len(target)))
for i in range(n_iter):
    params_print = jax.nn.softplus(params)
    (loss, y_ts), grads = loss_and_grad(params_softplus)
    if jnp.isnan(loss):
        print('NaN loss')
        n_iter = i
        break
    
    updates, opt_state = optimizer.update(grads, opt_state, params_softplus)
    params_softplus = optax.apply_updates(params_softplus, updates)

    loss_ts.append(loss)
    solutions_ts[i] = y_ts
    if i % 1 == 0:
        print(
            f"Iter {i:02d} | loss={loss:.3e} | "
            f"L={params_print[0].item():.3f} m | "
            f"D={jax.device_get(jax.nn.softplus(params_softplus[1:])).flatten()} Pa*s | "
            f"grads={grads.squeeze()}"
        )

params_opt = jax.nn.softplus(params_softplus)
print(f"Optimal L: {params_opt[0].item():.3f} m | Optimal D: {jax.device_get(params_opt[1:])} Pa*s")

# Print loss and optimization evolution
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
    ts=ts,
    solutions_ts=solutions_ts,
    target=target,
    show=True,
    save=True,
    savepath=plots_folder/'optimization_animation.gif'
)


# =====================================================
# Robot simulation after optimization
# =====================================================

# Update robot with optimal parameters
L_opt = jnp.array([params_opt[0]])
D_opt = jnp.diag(params_opt[1:])
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

# Simulate the optimized robot
ts, q_ts, qd_ts = robot_opt.resolve_upon_time(
    q0=q0,
    qd0=qd0,
    u=u,
    t0=t0,
    t1=t1,
    dt=dt,
    skip_steps=skip_step,
    solver=solver,
    max_steps=None,
)
g_ee_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L_opt))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]

# Error
MSE = onp.mean((y_ts - target) ** 2)
print(f'MSE (after optimization) = {MSE:.4f} m^2')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label=f'y end effector')
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
    t_list = ts,   # shape (T,)
    q_list = q_ts, # shape (T, DOF)
    target = target,
    slider = True,
    animation = False,
    show = True
)