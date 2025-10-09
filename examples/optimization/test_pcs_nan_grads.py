# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports and setup
import jax
import jax.numpy as jnp
import numpy as onp
from diffrax import Tsit5
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from IPython.display import HTML
from functools import partial
from soromox.systems.pcs import PCS

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
        return HTML(ani.to_jshtml())

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
        return HTML("Slider animation not implemented in HTML format. Use matplotlib directly to view the slider.")  # slider cannot be converted to HTML


# =====================================================
# Robot simulation before optimization
# =====================================================

# End effector target
target = jnp.array([0.0, 0.0, 0.2])

# Initialize robot
N = 1                     # segments
L = 1e-1 * jnp.ones((N,)) # segments length
L_max = jnp.sum(L)
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
q0 = jnp.tile(jnp.array([0.0, eps, 0.0, 0.5, 0.0, 0.0]), N) # ! eps is required to avoid nan gradients
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = 0.0
t1 = 10
dt = 1e-4
skip_step = 100
solver = Tsit5()
max_steps = int(1e7)

# ###################################
# SANDBOX
# ###################################
# from soromox.systems.utils import gauss_quadrature as gq
# from soromox.systems.utils import scale_gaussian_quadrature as sgq

# L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), L]))

# xs,ws,_ = gq(5,a=0.0,b=1.0)
# print(f'xs ......: {xs}')
# Xs_scaled, Ws_scaled = sgq(
#                 xs, ws, L_cum[0], L_cum[0 + 1]
#             )
# print(f'xs scaled: {Xs_scaled}')

# ###################################
# func = robot._coriolis_full_matrix
# func = robot._gravitational_force_full

# fwd_jac = jax.jacfwd(func)
# jac = fwd_jac(q0)
# print(jac)

# def pseudo_loss(q):
#     C = func(q)
#     J = jnp.sum(C)
#     return J
# val_and_grad = jax.value_and_grad(pseudo_loss)
# val, grad = val_and_grad(q0)
# print()
# print(f'val: {val}')
# print(f'grad: {grad}')
# exit()
# ###################################
# ###################################

# # Simulate robot
# ts, q_ts, _ = robot.resolve_upon_time(
#     q0 = q0, 
#     qd0 = qd0,
#     u = u, 
#     t0 = t0, 
#     t1 = t1, 
#     dt = dt, 
#     skip_steps = skip_step, 
#     solver = solver,
#     max_steps = max_steps
# )

# g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,L_max)
# ee_ts = g_ee_ts[:,:3,-1]

# # Plot results
# plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
# plt.plot(ts, ee_ts[:,1], color='r', label='y end effector')
# plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
# plt.axhline(target[0], color='b', linestyle='--', label='x target')
# plt.axhline(target[1], color='r', linestyle='--', label='y target')
# plt.axhline(target[2], color='g', linestyle='--', label='z target')
# plt.grid(True)
# plt.title('End effector position (before optimization)')
# plt.legend()
# plt.show()

# curve_i = draw_robot_curve(
#     batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
#     L_max = L_max,
#     q = q_ts[0,:],
# )
# x_i = curve_i[:,0]
# y_i = curve_i[:,1]
# z_i = curve_i[:,2]

# curve_f = draw_robot_curve(
#     batched_forward_kinematics=jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
#     L_max = L_max,
#     q = q_ts[-1,:],
# )
# x_f = curve_f[:,0]
# y_f = curve_f[:,1]
# z_f = curve_f[:,2]

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot(x_i,y_i,z_i, linewidth=3, label='initial')
# ax.plot(x_f,y_f,z_f, linewidth=3, label='final')
# ax.scatter(target[0],target[1],target[2], color='r')
# ax.set_xlim(-jnp.linalg.norm(robot.L)*1.5, jnp.linalg.norm(robot.L)*1.5)
# ax.set_ylim(-jnp.linalg.norm(robot.L)*1.5, jnp.linalg.norm(robot.L)*1.5)
# ax.set_zlim(0, jnp.linalg.norm(robot.L)*3)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Before optimization')
# ax.legend()
# plt.show()

# animate_robot_matplotlib(
#     robot = robot,
#     t_list = ts,   # shape (T,)
#     q_list = q_ts, # shape (T, DOF)
#     target = target,
#     slider = True,
#     animation = False,
# )


# =====================================================
# Optimization
# =====================================================

# Loss function
def Loss(L_softplus):
    L = jax.nn.softplus(L_softplus)               # get real, positive L
    robot_updated = robot.update_params({"L": L}) # update robot

    # simulation
    ts, q_ts, _ = robot_updated.resolve_upon_time(
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
    ee_final = robot_updated.forward_kinematics(q_ts[-1,:],L_max)[:3,-1]
    J = (ee_final-target).T @ (ee_final-target)

    # # test forward kinematics
    # ee_final = robot_updated.forward_kinematics(q0,L_max)[:3,-1]
    # J = (ee_final-target).T @ (ee_final-target)

    # # test forward dynamics
    # yd = robot_updated.forward_dynamics(
    #     t=0,
    #     y=jnp.concatenate([q0, qd0])
    # )
    # J = 1e-6 * jnp.sum(yd) ** 2

    return J, (q_ts, ts)

L = robot.L
L_softplus = L + jnp.log1p(-jnp.exp(-L)) # inverse softplus

from jax import value_and_grad
loss_and_grads = value_and_grad(Loss, has_aux=True)
(loss, (q_ts, ts)), grads = loss_and_grads(L_softplus)
print(f'loss = {loss}')
print(f'grads = {grads}')

# Plot initial and final configurations
curve_i = draw_robot_curve(
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
    L_max = L_max,
    q = q_ts[0,:],
)
x_i = curve_i[:,0]
y_i = curve_i[:,1]
z_i = curve_i[:,2]

curve_f = draw_robot_curve(
    batched_forward_kinematics=jax.vmap(robot.forward_kinematics, in_axes=(None, 0)),
    L_max = L_max,
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
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Before optimization')
ax.legend()
plt.show()

# Plot strains
fig, ax = plt.subplots(3, 2, figsize=(11,9), sharex=True)

ax[0,0].plot(ts, q_ts[:,0], label='q1')
ax[0,0].grid(True)
ax[0,0].legend()

ax[1,0].plot(ts, q_ts[:,1], label='q2')
ax[1,0].grid(True)
ax[1,0].legend()

ax[2,0].plot(ts, q_ts[:,2], label='q3')
ax[2,0].set_xlabel('t [s]')
ax[2,0].grid(True)
ax[2,0].legend()

ax[0,1].plot(ts, q_ts[:,3], label='q4')
ax[0,1].grid(True)
ax[0,1].legend()

ax[1,1].plot(ts, q_ts[:,4], label='q5')
ax[1,1].grid(True)
ax[1,1].legend()

ax[2,1].plot(ts, q_ts[:,5], label='q6')
ax[2,1].set_xlabel('t [s]')
ax[2,1].grid(True)
ax[2,1].legend()

plt.show()

# Animation
animate_robot_matplotlib(
    robot = robot,
    t_list = ts,   # shape (T,)
    q_list = q_ts, # shape (T, DOF)
    target = target,
    slider = True,
    animation = False,
)