import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import time
import pickle


# =====================================================
# Inverse softplus
# =====================================================
def InverseSoftplus(x):
    """
    Inverse softplus function.
    """
    return jnp.log(jnp.exp(x)-1)


# =====================================================
# Function for plotting loss function surface
# =====================================================
def PlotLoss(Loss : Callable,
             param1_range : tuple | Array,
             param2_range : tuple | Array,
             N1 = 25,
             N2 = 25,
             chunk_size : int = None,
             min_softplus : tuple = None,
             savepath = None
) -> tuple[Array, Array, Array]:
    """
    Plots 3D surface the loss as a function of the parameters. Works for 2 parameters.

    Args:
      Loss: 
        Loss function Loss(params) -> (J, ...). params is a (2,) vector.
      param1_range: 
        (min1, max1) tuple or array/sequence of values for first parameter.
      param2_range: 
        (min2, max2) tuple or array/sequence of values for second parameter.
      N1,N2: 
        Number of discretization points for each parameter axis, if param*_range are (min*, max*).
        Default: 25.
      chunk_size: 
        If not None, evaluates the loss on blocks of the parameter's grid of dimension chunk_size. 
        If None, evaluates the loss on the entire grid all at once.
      min_softplus:
        (Just for rescaling the axes) Not None when the parameters are re-parametrized with softplus in 
        the loss. This argument should be a tuple (par1_min, par2_min) equal to the minimum threshold 
        for each parameter (set (0,0) if a pure softplus was used).
      savepath:
        path where saving the plots. Must be in form .../savefolder/'name_figure'. Saves all plots as png 
        and also 3D surface plot as binary file that can be re-loaded in python. If None, plots are not
        saved.

    Return:
      Z:
        loss function values.
      param_nan:
        array containing the values of the parameters that give nan. Each row is a pair
        of values (param1, param2) that gives nan in the simulation. Shape (N_nan, 2). 
      param_inf:
        array containing the values of the parameters that give + or - inf. Each row is a pair
        of values (param1, param2) that gives inf in the simulation. Shape (N_inf, 2). 
    """

    # Prepare x,y grid
    if len(param1_range) > 2:
        x = jnp.array(param1_range)
    else:
        p1_min, p1_max = param1_range
        x = jnp.linspace(p1_min, p1_max, N1)

    if len(param2_range) > 2:
        y = jnp.array(param2_range)
    else:
        p2_min, p2_max = param2_range
        y = jnp.linspace(p2_min, p2_max, N2)

    X, Y = jnp.meshgrid(x, y, indexing='xy') # indexing='xy' => shape (len(y), len(x))
    n_points = X.size
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # shape (n_points, 2): all the points in the grid stacked

    start = time.perf_counter()
    # Jit and vmap loss function
    Loss_jit = jax.jit(lambda p: Loss(p)[0])
    Loss_vmapped = jax.vmap(Loss_jit, in_axes=0)

    # Evaluate loss function for all points in the grid
    if chunk_size is None or chunk_size >= n_points:
        z_flat = Loss_vmapped(points) # shape (n_points,)
        z_flat = onp.array(jax.device_get(z_flat))
    else:
        z_parts = []
        i = 0
        k = 1
        while i < n_points:
            print(f'Starting block {k}...')
            _start = time.perf_counter()
            i2 = min(i + chunk_size, n_points)
            chunk = points[i:i2]
            z_chunk = Loss_vmapped(chunk)
            z_chunk = onp.array(jax.device_get(z_chunk))
            z_parts.append(z_chunk)
            i = i2
            _end = time.perf_counter()
            print(f'Finished block {k} in {_end - _start} s')
            k += 1
        z_flat = onp.concatenate(z_parts, axis=0)

    end = time.perf_counter()
    print(f'Time for computing loss: {end-start} s')

    # Convert z_flat into mesh form 
    Z = z_flat.reshape(X.shape)   # shape (n_points) --> shape (len(y), len(x))
    X = onp.array(jax.device_get(X))
    Y = onp.array(jax.device_get(Y))
    if min_softplus is not None:
        par1_min, par2_min = min_softplus
        X = jax.nn.softplus(X) + par1_min
        Y = jax.nn.softplus(Y) + par2_min

    # Warning if some simulation gave NaN or inf
    param_nan = []
    param_inf = []
    if onp.any(onp.isnan(Z)) or onp.any(onp.isinf(onp.abs(Z))):
        N_nan = onp.sum(onp.isnan(Z))
        N_inf = onp.sum(onp.isinf(onp.abs(Z)))
        print(f'Warning: {N_nan} NaN and {N_inf} inf detected!')
        # plot nan and inf regions
        pos_nan = onp.argwhere(onp.isnan(Z))
        pos_inf = onp.argwhere(onp.isinf(Z))
        x_nan = X[pos_nan[:,0],pos_nan[:,1]]
        y_nan = Y[pos_nan[:,0],pos_nan[:,1]]
        x_inf = X[pos_inf[:,0],pos_inf[:,1]]
        y_inf = Y[pos_inf[:,0],pos_inf[:,1]]
        param_nan = onp.column_stack((x_nan, y_nan))
        param_inf = onp.column_stack((x_inf, y_inf))
        plt.figure()
        plt.scatter(x_nan, y_nan, c='r', label='NaN locations')
        plt.scatter(x_inf, y_inf, c='b', label='Inf locations')
        plt.grid(True)
        plt.xlabel('param 1')
        plt.ylabel('param 2')
        plt.xlim([onp.min(X), onp.max(X)])
        plt.ylim([onp.min(Y), onp.max(Y)])
        plt.legend()
        plt.title(f'Detected {N_nan}/{n_points} NaN and {N_inf}/{n_points} Inf')
        if savepath is not None:
          plt.savefig(f'{savepath}_nanlocations', bbox_inches='tight')
        
    # 3D plotting
    z_lower = onp.min(Z[onp.isfinite(Z) & (Z>-1e10)])
    z_upper = onp.max(Z[onp.isfinite(Z) & (Z<1e10)])

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(onp.array(X), onp.array(Y), Z, cmap='viridis', edgecolor='none', vmin=z_lower, vmax=z_upper)
    ax.set_xlabel('parameter 1')
    ax.set_ylabel('parameter 2')
    ax.set_zlabel('J')
    ax.set_xlim([onp.min(X), onp.max(X)])
    ax.set_ylim([onp.min(Y), onp.max(Y)])
    ax.set_zlim([z_lower, z_upper])
    ax.set_title('Loss function surface plot')
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
    if savepath is not None:
      plt.savefig(savepath, bbox_inches='tight')
      with open(savepath, 'wb') as f:
          pickle.dump(plt.gcf(), f)
    plt.show()

    return Z, onp.array(param_nan), onp.array(param_inf)


# =====================================================
# Function for plotting evolution of the solution(s) during optimization
# =====================================================
def plot_optimiz_evolution(
        ts : Array, 
        solutions : Array, 
        targets : Array = None, 
        show : bool = True, 
        savepath = None
):
    """
    Plots the evolution of the solutions sol1(t), sol2(t), ... during optimization.

    Args
    ----
    ts : Array
      Time vector. Shape (N_timeinstants,)
    solutions : Array
      Array with the solution(s) at each iteration. Shape (N_solutions, N_iterations, N_timeinstants)
    targets : Array, optional
      Target solution(s). Shape (N_solutions, N_timeistants)
    show : bool
      True for showing the animation (default: True).
    savepath :
      Path where saving animation, in form: folder/'title_animation.gif' (default: None).
      If None, animation is not saved.
    """
    # Initial checks
    if len(solutions.shape) != 3:
      N_sol = 1
      solutions = solutions[None, :, :]
      if targets is not None:
        targets = targets[None, :]
    else:
      N_sol = solutions.shape[0]

    flag = False
    if targets is None:
      targets = onp.full((N_sol, solutions.shape[-1]), onp.nan)
      flag = True

    # Limits for the y axis
    y_lower = onp.zeros(N_sol)
    y_upper = onp.ones(N_sol)
    for ii in range(N_sol):
      solutions_ii = solutions[ii, :, :]
      target_ii = targets[ii, :]
      if flag:
        y_lower[ii] = onp.min([solutions_ii[onp.isfinite(solutions_ii) & (solutions_ii>-1e6)].min()])
        y_upper[ii] = onp.max([solutions_ii[onp.isfinite(solutions_ii) & (solutions_ii<1e6)].max()])
      else:
        y_lower[ii] = onp.min([solutions_ii[onp.isfinite(solutions_ii) & (solutions_ii>-1e6)].min(), target_ii[onp.isfinite(target_ii)].min()])
        y_upper[ii] = onp.max([solutions_ii[onp.isfinite(solutions_ii) & (solutions_ii<1e6)].max(), target_ii[onp.isfinite(target_ii)].max()])

    # Create slider animation   
    if show:
      fig, ax = plt.subplots(N_sol, 1, figsize=(8, 6), sharex=True)
      if N_sol==1:
        ax = [ax]
      plt.subplots_adjust(bottom=0.15)
      
      ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.03])  # [left, bottom, width, height]

      def update_plot(frame_idx):
          for ii in range(N_sol):
            ax[ii].cla()  # clear current axes
            if ii == 0:
              ax[ii].set_title(f"Optimization (iteration n. {frame_idx})")
            else:
              ax[ii].set_title('')
            if ii == N_sol-1:
              ax[ii].set_xlabel("t [s]")
            else:
              ax[ii].set_xlabel('')
            ax[ii].plot(ts, solutions[ii, frame_idx, :], color='r', label=f'solution(t)')
            ax[ii].plot(ts, targets[ii, :], color='r', linestyle='--', label='target(t)')
            ax[ii].set_xlim(ts[0], ts[-1])
            ax[ii].set_ylim(y_lower[ii], y_upper[ii])
            ax[ii].set_ylabel(f"sol_{ii+1}")
            ax[ii].grid(True)
            ax[ii].legend()

            fig.canvas.draw_idle()

      slider = Slider(
          ax=ax_slider,
          label="Iter.",
          valmin=0,
          valmax=solutions.shape[1] - 1,
          valinit=0,
          valstep=1,
      )
      slider.on_changed(update_plot)
      update_plot(0)  # initial plot
      plt.show()
    
    # Create animation to be saved
    if savepath is not None:        
        fig, ax = plt.subplots(N_sol, 1, figsize=(8, 6), sharex=True)
        if N_sol==1:
           ax = [ax]

        line_sim = []
        line_tar = []
        for ii in range(N_sol):
          line_sim.append(ax[ii].plot([], [], color='r', label=f'solution(t)')[0])
          line_tar.append(ax[ii].plot([], [], color='r', linestyle='--', label='target(t)')[0])
          ax[ii].set_xlim(ts[0], ts[-1])
          ax[ii].set_ylim(y_lower[ii], y_upper[ii])
          ax[ii].set_ylabel(f'sol_{ii+1}')
          ax[ii].grid(True)
          ax[ii].legend()
        ax[0].set_title(f"Optimization (iteration n. 0)")
        ax[-1].set_xlabel("t [s]")
        fig.tight_layout()

        def init():
            for ii in range(N_sol):
              line_sim[ii].set_data([], [])
              line_tar[ii].set_data(ts, targets[ii,:])
            return line_sim + line_tar

        def update(frame_idx):
            for ii in range(N_sol):
              line_sim[ii].set_data(ts, solutions[ii, frame_idx, :])
              if ii == 0:
                ax[0].set_title(f"Optimization (iteration n. {frame_idx})")
            return line_sim + line_tar
        
        ani = FuncAnimation(
            fig,
            update,
            frames=solutions.shape[1],
            init_func=init,
            blit=True
        )

        ani.save(savepath, writer="pillow", fps=solutions.shape[1]/10)
        plt.close(fig)