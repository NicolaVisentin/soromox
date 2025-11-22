# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
from jax import numpy as jnp
import optax
import matplotlib.pyplot as plt
from soromox.systems import pendulum
from soromox.systems.system_state import SystemState

jax.config.update("jax_enable_x64", True)  # double precision


num_links = 2
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
    "D": jnp.diag(jnp.array([50, 20]))
}

# define initial configuration
q0 = jnp.array([-jnp.pi * 7/11, jnp.pi * 7/13])

# set simulation parameters
dt = 1e-3        # time step
t0 = 0.0
t1 = 15.0
save_dt = 100*dt # time resolution for saving results


if __name__ == "__main__":
    target = jnp.array([0.0, -2.0])
    # Instantiate the pendulum model directly
    robot = pendulum.Pendulum(params)

    # initialize velocities and actuation
    qd0 = jnp.zeros_like(q0)  # initial velocities for simulation
    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    u = jnp.zeros_like(q0)  # torques (actuation)

    # Integrate using the model's built-in solver
    sim_out = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=dt,
        save_dt=save_dt,
    )

    # Extract results
    ts_out = sim_out.t
    q_ts, qd_ts = jnp.split(sim_out.y, 2, axis=1)
    

    # =====================================================
    # End-effector position upon time
    # =====================================================
    chi_ee_ts = jax.vmap(robot.forward_kinematics_tips,)(q_ts)[:, -1, :]

    # plot end-effector position vs time
    plt.figure()
    plt.plot(ts_out, chi_ee_ts[:, 1], label="End-effector x [m]", color='b')
    plt.plot(ts_out, chi_ee_ts[:, 2], label="End-effector y [m]", color='r')
    plt.axhline(target[0], label='target x', linestyle='--', color='b')
    plt.axhline(target[1], label='target y', linestyle='--', color='r')
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts_out, cmap="viridis")
    plt.scatter(target[0], target[1], color='r', label='target')
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.legend()
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # OPTIMIZATION
    # =====================================================
    def Loss(L_softplus):
        L = jax.nn.softplus(L_softplus)
        robot_updated = robot.update_params({"L": L})

        # simulation
        sim_out = robot_updated.rollout_to(
            initial_state=initial_state,
            u=u,
            t1=t1,
            solver_dt=dt,
            save_dt=save_dt,
            max_steps=int(1e5)
        )
        q_ts, _ = jnp.split(sim_out.y, 2, axis=1)
        ee_final = robot_updated.forward_kinematics_tips(q_ts[-1,:])[-1, 1:]
        J = (ee_final-target).T @ (ee_final-target)

        # # forward dynamics
        # yd = robot_updated.forward_dynamics(
        #     t=0,
        #     y=jnp.concatenate([q0, qd0])
        # )
        # J = jnp.sum(yd) ** 2

        return J

    L = robot.L
    L_softplus = L + jnp.log1p(-jnp.exp(-L)) # inverse softplus

    # from jax import grad
    # grads = grad(Loss)(L)
    # print(Loss(L_softplus))
    # print(grads)
    # exit()

    optimizer = optax.sgd(learning_rate=0.05) 
    opt_state = optimizer.init(L_softplus)          

    print('Starting optimization...')
    loss_and_grad = jax.value_and_grad(Loss)
    for i in range(50):
        L_softplus_iter = L_softplus
        L_iter = jax.nn.softplus(L_softplus)
        loss, grads = loss_and_grad(L_softplus)
        updates, opt_state = optimizer.update(grads, opt_state, L_softplus)
        L_softplus = optax.apply_updates(L_softplus, updates)

        print(f"Iter {i:02d} | loss={loss:.3e}| grads={grads} | L1={L_iter[0]:.3f} | L2={L_iter[1]:.3f} | L1_softplus={L_softplus_iter[0]:.3f} | L2_softplus={L_softplus_iter[1]:.3f}")

    L_opt = jax.nn.softplus(L_softplus)
    print("Optimal lengths:", L_opt)

    robot = robot.update_params({'L': L_opt})
    sim_out = robot.rollout_to(
            initial_state=initial_state,
            u=u,
            t1=t1,
            solver_dt=dt,
            save_dt=save_dt,
            max_steps=int(1e5)
        )
    ts_out = sim_out.t
    q_ts, _ = jnp.split(sim_out.y, 2, axis=1)
    ee_final = robot.forward_kinematics_tips(q_ts[-1,:])[-1, 1:]
    print(f'ee: {ee_final}')
    print(f'target: {target}')

    chi_ee_ts = jax.vmap(robot.forward_kinematics_tips,)(q_ts)[:, -1, :]

    # plot end-effector position vs time
    plt.figure()
    plt.plot(ts_out, chi_ee_ts[:, 1], label="End-effector x [m]", color='b')
    plt.plot(ts_out, chi_ee_ts[:, 2], label="End-effector y [m]", color='r')
    plt.axhline(target[0], label='target x', linestyle='--', color='b')
    plt.axhline(target[1], label='target y', linestyle='--', color='r')
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts_out, cmap="viridis")
    plt.scatter(target[0], target[1], color='r', label='target')
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.legend()
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()