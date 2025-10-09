import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.colors as colors


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import time


# Numerically stable inverse softplus
def InverseSoftplus(x):
    """
    Numerically stable inverse softplus function.
    """
    return x + jnp.log1p(-jnp.exp(-x))


# Function for plotting loss function surface
def PlotLoss(Loss,
             param1_range,
             param2_range,
             N1=25,
             N2=25,
             chunk_size=None
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

    Return:
      param_nan:
        array containing the values of the parameters that give nan. Each row is a pair
        of values (param1, param2) that gives nan in the simulation. Shape (N_nan, 2). 
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

    # Warning if some simulation gave NaN
    param_nan = []
    if onp.any(onp.isnan(Z)):
        N_nan = onp.sum(onp.isnan(Z))
        print(f'Warning: {N_nan} NaN detected!')
        # plot nan regions
        pos_nan = onp.argwhere(onp.isnan(Z))
        x_nan = X[pos_nan[:,0],pos_nan[:,1]]
        y_nan = Y[pos_nan[:,0],pos_nan[:,1]]
        param_nan = onp.column_stack((x_nan, y_nan))
        plt.figure()
        plt.scatter(jax.nn.softplus(x_nan), jax.nn.softplus(y_nan), label='NaN locations')
        plt.grid(True)
        plt.xlabel('param 1')
        plt.ylabel('param 2')
        plt.title(f'Detected {N_nan}/{n_points} NaN')
        
    # 3D plotting
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(onp.array(jax.nn.softplus(X)), onp.array(jax.nn.softplus(Y)), Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('parameter 1')
    ax.set_ylabel('parameter 2')
    ax.set_zlabel('J')
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
    ax.set_title('Loss function surface plot')
    plt.show()

    return onp.array(jax.nn.softplus(param_nan))