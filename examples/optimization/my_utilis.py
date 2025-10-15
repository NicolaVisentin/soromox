import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import numpy as onp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import time


# Inverse softplus
def InverseSoftplus(x):
    """
    Inverse softplus function.
    """
    return jnp.log(jnp.exp(x)-1)


# Function for plotting loss function surface
def PlotLoss(Loss,
             param1_range,
             param2_range,
             N1=25,
             N2=25,
             chunk_size=None,
             min_softplus=None
):
    """
    Plots 3D surface the loss as a function of the parameters. Works for 2 parameters.

    Args:
      Loss: 
        Loss function Loss(params) -> (J, ...). params is a (2,) vector.
      param1_range: 
        (min1, max1) or array/sequence of values for first parameter.
      param2_range: 
        (min2, max2) or array/sequence of values for second parameter.
      N1,N2: 
        Number of discretization points for each parameter axis, if param*_range are (min*, max*).
      chunk_size: 
        If not None, evaluates the loss on blocks of the parameter's grid of dimension chunk_size. 
        If None, evaluates the loss on the entire grid all at once.
      min_softplus:
        (Just for rescaling the axes) Not None if the parameters are re-parametrized with softplus in 
        the loss. This argument should be a tuple (par1_min, par2_min) equal to the minimum threshold 
        for each parameter (set (0,0) if a pure softplus was used).

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
    plt.show()

    return Z, onp.array(param_nan), onp.array(param_inf)
