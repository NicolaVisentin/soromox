# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports and setup
import numpy as onp
from numpy import correlate
from scipy.signal import find_peaks
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


# =====================================================
# Target creation
# =====================================================

# End effector target (want to match the period T along y coordinate)
t0 = 0.0
t1 = 10.0
dt = 1e-3
t_target = jnp.arange(t0, t1, dt)
c = 0.5
k = 10
wd = jnp.sqrt(4*k-c**2)/2
T_target = 2*jnp.pi/wd
target = 0.75 * jnp.exp(-c/2 * t_target) * jnp.cos(wd * t_target)


# =====================================================
# Robot simulation before optimization
# =====================================================

# Initialize robot
N = 1                     # segments
L = 3e-1 * jnp.ones((N,)) # segments length
E = 2e3 * jnp.ones((N,))  # elastic modulus
parameters = {
    "p0": jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0, 0.0, 0.0]),
    "L": L,
    "r": 2e-2 * jnp.ones((N,)),
    "rho": 1070 * jnp.ones((N,)),
    "g": jnp.array([0.0, 0.0, 9.81]),
    "E": E,
    "G": 1e3 * jnp.ones((N,)),
    "D": 1e-3 * jnp.diag((jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), N, axis=0) * L[:, None]).flatten())
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
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]

# Estimate period T
autocorr = correlate(y_ts, y_ts, mode='full')
autocorr = autocorr[autocorr.size//2:]
peaks, _ = find_peaks(autocorr)
T = ts[peaks[0]]
print(f'Before optimization: T_target = {T_target:.4f} s | T = {T:.4f} s | Error = {onp.abs(T_target - T):.4f} s')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label=f'y end effector (T={T:.4f} s)')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.plot(t_target, target, color='r', linestyle='--', label=f'target y (T={T_target:.4f} s)')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.grid(True)
plt.title(f'End effector position (before optimization, L={L[0]:.3f} m, E={E[0]:.0f} Pa)')
plt.legend()
plt.show()

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

# Function for computing normalized differentiable autocorrelation
@jax.jit
def autocorr_norm(signal):
    y = signal - jnp.mean(signal)
    R_yy = jnp.correlate(y, y, mode='full')
    R_yy = R_yy[R_yy.size // 2:]
    R_signal = R_yy / jnp.max(R_yy) # normalize
    return R_signal

R_target = autocorr_norm(target)
R_yy = autocorr_norm(y_ts)

plt.figure()
plt.plot(dt*jnp.arange(len(R_target)), R_target, label=f'y target (T={T_target:.4f} s)')
plt.plot(dt*jnp.arange(len(R_yy)), R_yy, label=f'y simulation (T={T:.4f} s)')
plt.xlabel(r'$\tau$ [s]')
plt.ylabel(r'$R_{yy}$')
plt.title('Autocorrelation (before optimization)')
plt.legend()
plt.grid(True)
plt.show()

# Loss function (second try: FFT)
def Loss(params_softplus):
    params = jax.nn.softplus(params_softplus)             # get real, positive parameters
    L, E = params                                         # extract optimization parameters
    robot_updated = robot.update_params({'L': L, 'E': E}) # update robot

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
    g_ee_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
    ee_ts = g_ee_ts[:,:3,-1]
    y_ts = ee_ts[:,1]

    # period estimation
    y = y_ts - jnp.mean(y_ts) # remove DC, so no peak at f=0 Hz

    win = jnp.hanning(len(y)) # apply Hanning window, to reduce leakage
    yw = y * win

    N_pad = 4*len(y)
    Y = jnp.fft.rfft(yw, n=N_pad)      # FFT. Zero padding to increase frequency resolution
    f = jnp.fft.rfftfreq(N_pad, d=dt)  # new frequecy axis

    eps = jnp.finfo(jnp.float64).eps  # machine epsilon
    G_yy = jnp.abs(Y)**2 + eps # compute power spectrum. Add eps for numerical stability
    G_yy = jnp.log(G_yy)       # use log to help softmax

    f_nyq = 0.5 / dt  # filter high frequencies (above Nyquist)
    filter = f<=f_nyq
    G_yy = jnp.where(filter, G_yy, -1e12)

    weights = jax.nn.softmax(G_yy) # find peaks on the power spectrum

    f_hat = jnp.sum(weights*f) # estimate frequency basing on weighted sum (can't use argmax because of differentiability)
    T_hat = 1.0 / f_hat

    J = 1e1 * jnp.mean((T_hat - T_target) ** 2) + 1e-3 * E[0]**2 # add L2 regularization on E to penalize it a bit

    return J

params = jnp.array([L, E])
params_softplus = params + jnp.log1p(-jnp.exp(-params)) # inverse softplus

# # !! Check gradients computation !!
# loss_and_grad = jax.jit(jax.value_and_grad(Loss))
# loss, grads = loss_and_grad(params_softplus)
# print(loss, grads)
# exit()

# Setup optimizer
optimizer = optax.sgd(learning_rate=5e-2)   # use SGD
opt_state = optimizer.init(params_softplus) # initialize optimizer

# Optimization iterations
print('Starting optimization...')
loss_and_grad = jax.jit(jax.value_and_grad(Loss))
loss_ts = []
n_iter = 10
for i in range(n_iter):
    loss, grads = loss_and_grad(params_softplus)
    updates, opt_state = optimizer.update(grads, opt_state, params_softplus)
    params_softplus = optax.apply_updates(params_softplus, updates)

    loss_ts.append(loss)
    if i % 1 == 0:
        print(f"Iter {i:02d} | loss={loss:.3e} | L={jax.nn.softplus(params_softplus[0]).item():.3f} m | E={jax.nn.softplus(params_softplus[1]).item():.0f} Pa | grads={grads.squeeze()}")

params_opt = jax.nn.softplus(params_softplus)
print(f"Optimal L: {params_opt[0].item():.3f} m | Optimal E: {params_opt[1].item():.0f} Pa")

plt.figure()
plt.plot(range(n_iter), loss_ts)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


# =====================================================
# Robot simulation after optimization
# =====================================================

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": params_opt[0], "E": params_opt[1]})

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
g_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts,jnp.sum(L))
ee_ts = g_ee_ts[:,:3,-1]
y_ts = ee_ts[:,1]

# Estimate period T
autocorr = correlate(y_ts, y_ts, mode='full')
autocorr = autocorr[autocorr.size//2:]
peaks, _ = find_peaks(autocorr)
T = ts[peaks[0]]
print(f'After optimization: T_target = {T_target:.4f} s | T = {T:.4f} s | Error = {onp.abs(T_target - T):.4f} s')

# Plot results
plt.figure()
plt.plot(ts, ee_ts[:,0], color='b', label='x end effector')
plt.plot(ts, ee_ts[:,1], color='r', label=f'y end effector (T={T:.4f} s)')
plt.plot(ts, ee_ts[:,2], color='g', label='z end effector')
plt.plot(t_target, target, color='r', linestyle='--', label=f'target y (T={T_target:.4f} s)')
plt.xlabel('t [s]')
plt.ylabel('pos [m]')
plt.grid(True)
plt.title(f'End effector position (after optimization, L={params_opt[0].item():.3f} m, E={params_opt[1].item():.0f} Pa)')
plt.legend()
plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = ts,   # shape (T,)
    q_list = q_ts, # shape (T, DOF)
    target = target,
    slider = True,
    animation = False,
    show = True
)

R_yy = autocorr_norm(y_ts)

plt.figure()
plt.plot(dt*jnp.arange(len(R_target)), R_target, label=f'y target (T={T_target:.4f})')
plt.plot(dt*jnp.arange(len(R_yy)), R_yy, label=f'y simulation (T={T:.4f})')
plt.xlabel(r'$\tau$ [s]')
plt.ylabel(r'$R_{yy}$')
plt.title('Autocorrelation (after optimization)')
plt.legend()
plt.grid(True)
plt.show()