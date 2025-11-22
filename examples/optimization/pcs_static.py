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
from soromox.systems.system_state import SystemState

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

    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = jnp.sum(robot.L)

    width = jnp.linalg.norm(robot.L) * 3
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if slider:
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    if animation:
        (line,) = ax.plot([], [], [], lw=4, color="blue")
        if target is not None:
            ax.scatter(target[0],target[1],target[2], color='r', label='target')
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve = draw_robot_curve(batched_forward_kinematics, L_max, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            line.set_3d_properties(curve[:, 2])
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
            ax.cla()  # clear current axes
            if target is not None:
                ax.scatter(target[0],target[1],target[2], color='r', label='target')
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

# Folder for plots and videos
curr_folder = Path(__file__).parent
plots_folder = curr_folder/'plots and videos'/Path(__file__).stem
plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Robot simulation before optimization
# =====================================================

# End effector target
target = jnp.array([0.0, 0.0, 0.2])

# Initialize robot
N = 1                     # segments
L = 1e-1 * jnp.ones((N,)) # segments length
parameters = {
    "p0": jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0, 0.0, 0.0]),
    "L": L,
    "r": 2e-2 * jnp.ones((N,)),
    "rho": 1070 * jnp.ones((N,)),
    "g": jnp.array([0.0, 0.0, 9.81]),
    "E": 2e3 * jnp.ones((N,)),
    "G": 1e3 * jnp.ones((N,)),
    "D": 1e-3 * jnp.diag((jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), N, axis=0) * L[:, None]).flatten())
}

robot = PCS(
    num_segments = N,
    params = parameters,
    order_gauss = 5
)

# Simulation parameters
eps = jnp.finfo(jnp.float64).eps
q0 = jnp.tile(jnp.array([0.0, 3.0, 0.0, 1.0, 0.0, 0.0]), N) # ! nan gradients if q[0]=q[1]=q[2]=0.0
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = 0.0
t1 = 5
dt = 1e-3
save_dt = 100*dt
solver = Tsit5()
max_steps = int(1e6)
initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

# Simulate robot
sim_out = robot.rollout_to(
    initial_state=initial_state,
    u = u, 
    t1 = t1, 
    solver_dt = dt, 
    save_dt = save_dt, 
    solver = solver,
    max_steps = max_steps
)
ts = sim_out.t
q_ts, _ = jnp.split(sim_out.y, 2, axis=1)
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]

error = jnp.linalg.norm(g_ee_ts[-1,0:3,3]-target)
print(f'Fianl error (before optimization): {error:.6f} m')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label='y end effector')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.axhline(target[0], color='b', linestyle='--', label='x target')
plt.axhline(target[1], color='r', linestyle='--', label='y target')
plt.axhline(target[2], color='g', linestyle='--', label='z target')
plt.grid(True)
plt.title('End effector position (before optimization)')
plt.legend()
plt.savefig(plots_folder/'End effector position (before optimization)')
#plt.show()

curve_i = draw_robot_curve(
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
    L_max = jnp.sum(L),
    q = q_ts[0,:],
)
x_i = curve_i[:,0]
y_i = curve_i[:,1]
z_i = curve_i[:,2]

curve_f = draw_robot_curve(
    batched_forward_kinematics=jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
    L_max = jnp.sum(L),
    q = q_ts[-1,:],
)
x_f = curve_f[:,0]
y_f = curve_f[:,1]
z_f = curve_f[:,2]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(x_i,y_i,z_i, linewidth=3, label='initial')
ax.plot(x_f,y_f,z_f, linewidth=3, label='final')
ax.scatter(target[0],target[1],target[2], color='r')
ax.set_xlim(-jnp.linalg.norm(robot.L)*1.5, jnp.linalg.norm(robot.L)*1.5)
ax.set_ylim(-jnp.linalg.norm(robot.L)*1.5, jnp.linalg.norm(robot.L)*1.5)
ax.set_zlim(0, jnp.linalg.norm(robot.L)*3)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Before optimization')
ax.legend()
plt.savefig(plots_folder/'Initial and final configurations (before optimization)')
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
def Loss(L_softplus):
    L = jax.nn.softplus(L_softplus)               # get real, positive L
    robot_updated = robot.update_params({"L": L}) # update robot

    # simulation
    sim_out = robot_updated.rollout_to(
        initial_state=initial_state,
        u = u, 
        t1 = t1, 
        solver_dt = dt, 
        save_dt = save_dt, 
        solver = solver,
        max_steps = max_steps
    )                                              # simulate
    q_ts, _ = jnp.split(sim_out.y, 2, axis=1)
    ee_final = robot_updated.forward_kinematics(q_ts[-1,:],jnp.sum(L))[:3,-1]
    P = jnp.diag(jnp.array([1,1,3e3]))
    J = (ee_final-target).T @ P @ (ee_final-target)

    return J

L = robot.L
L_softplus = L + jnp.log1p(-jnp.exp(-L)) # inverse softplus

# # !! Check gradients computation !!
# loss_and_grad = jax.jit(jax.value_and_grad(Loss))
# loss, grads = loss_and_grad(L_softplus)
# print(loss, grads)
# exit()

# Setup optimizer
optimizer = optax.sgd(learning_rate=5e-3)  # use SGD
opt_state = optimizer.init(L_softplus)     # initialize optimizer

# Optimization iterations
print('Starting optimization...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss))
loss_ts = []
n_iter = 10
for i in range(n_iter):
    L_softplus_print = L_softplus
    L_print = jax.nn.softplus(L_softplus)
    loss, grads = loss_and_grad(L_softplus)
    updates, opt_state = optimizer.update(grads, opt_state, L_softplus)
    L_softplus = optax.apply_updates(L_softplus, updates)

    loss_ts.append(loss)
    if i % 1 == 0:
        print(f"Iter {i:02d} | loss={loss:.3e} | L_softplus={L_softplus_print} | L={L_print} | grads={grads}")

L_opt = jax.nn.softplus(L_softplus)
print("Optimal lengths:", L_opt)

plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(plots_folder/'Loss')
#plt.show()


# =====================================================
# Robot simulation after optimization
# =====================================================

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": L_opt})

# Simulate the optimized robot
sim_out = robot_opt.rollout_to(
    initial_state=initial_state,
    u=u,
    t1=t1,
    solver_dt=dt,
    save_dt=save_dt,
    solver=solver,
    max_steps=None,
)
ts = sim_out.t
q_ts, qd_ts = jnp.split(sim_out.y, 2, axis=1)
g_ee_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L_opt))
ee_ts = g_ee_ts[:,:3,-1]

error = jnp.linalg.norm(g_ee_ts[-1,0:3,3]-target)
print(f'Fianl error (after optimization): {error:.6f} m')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label='y end effector')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.axhline(target[0], color='b', linestyle='--', label='x target')
plt.axhline(target[1], color='r', linestyle='--', label='y target')
plt.axhline(target[2], color='g', linestyle='--', label='z target')
plt.grid(True)
plt.title('End effector position (after optimization)')
plt.legend()
plt.savefig(plots_folder/'End effector position (after optimization)')
#plt.show()

curve_i = draw_robot_curve(
    batched_forward_kinematics = jax.vmap(robot_opt.forward_kinematics, in_axes=(None, 0)),
    L_max = jnp.sum(L_opt),
    q = q_ts[0,:],
)
x_i = curve_i[:,0]
y_i = curve_i[:,1]
z_i = curve_i[:,2]

curve_f = draw_robot_curve(
    batched_forward_kinematics=jax.vmap(robot_opt.forward_kinematics, in_axes=(None, 0)),
    L_max = jnp.sum(L_opt),
    q = q_ts[-1,:],
)
x_f = curve_f[:,0]
y_f = curve_f[:,1]
z_f = curve_f[:,2]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(x_i,y_i,z_i, linewidth=3, label='initial')
ax.plot(x_f,y_f,z_f, linewidth=3, label='final')
ax.scatter(target[0],target[1],target[2], color='r')
ax.set_xlim(-jnp.linalg.norm(robot_opt.L)*1.5, jnp.linalg.norm(robot_opt.L)*1.5)
ax.set_ylim(-jnp.linalg.norm(robot_opt.L)*1.5, jnp.linalg.norm(robot_opt.L)*1.5)
ax.set_zlim(0, jnp.linalg.norm(robot_opt.L)*3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('After optimization')
ax.legend()
plt.savefig(plots_folder/'Initial and final configurations (after optimization)')
#plt.show()

animate_robot_matplotlib(
    robot = robot_opt,
    t_list = ts,   # shape (T,)
    q_list = q_ts, # shape (T, DOF)
    target = target,
    slider = True,
    animation = False,
)