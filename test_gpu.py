# test_gpu_vs_cpu_jax.py
import time
import jax
import jax.numpy as jnp
import numpy as np

# ================================================================
# FUNCTIONS
# ================================================================

def time_fn_on_device(fn, args, device, warmups=2, repeats=10):
    # Put args on the chosen device
    args_on_dev = [jax.device_put(a, device) for a in args]

    # Warm-up runs (to compile + cache)
    for _ in range(warmups):
        out = fn(*args_on_dev)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args_on_dev)
        # ensure completion
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

        t1 = time.perf_counter()
        times.append(t1 - t0)
        
    return np.mean(times), np.std(times), times

def make_test_fn(size, repeat_inner=1):
    # heavy operation: repeated matmul accumulate
    @jax.jit
    def heavy(a, b):
        # do repeat_inner successive matmuls (keeps graph bigger)
        x = a
        for _ in range(repeat_inner):
            x = jnp.dot(x, b)
        # reduce to scalar so result is used
        return jnp.sum(x)
    return heavy


# ================================================================
# MAIN
# ================================================================

def main():
    # Parameters
    SIZE = 5001         # matrices dimensions (SIZE x SIZE)
    REPEAT_INNER = 3    # how many times we want to perform matrix multiplication
    WARMUPS = 2         # number of warmups for jitted functions
    REPEATS = 6         # number of test (to have averaged results in terms of elapsed time)

    print("\n--- FINDING DEVICES ---\nAvailable devices:", jax.devices())
    devices = {d.platform: d for d in jax.devices()}  # keep one per platform

    if "gpu" not in devices:
        print("WARNING: no JAX GPU ('gpu') was detected. The test will only be performed on the CPU.")
    else:
        print("GPU detected: ", devices["gpu"])

    cpu_device = devices.get("cpu", jax.devices("cpu")[0])
    gpu_device = devices.get("gpu", None)

    # Create random inputs (on host)
    print(f"\nCreating random matrices {SIZE}x{SIZE} (on host)...")
    a = np.random.randn(SIZE, SIZE).astype(np.float32)
    b = np.random.randn(SIZE, SIZE).astype(np.float32)

    # Build jitted function (it will be compiled separately per device)
    heavy_fn = make_test_fn(SIZE, repeat_inner=REPEAT_INNER)

    print("\n--- TEST on CPU ---")
    cpu_mean, cpu_std, cpu_times = time_fn_on_device(heavy_fn, [a, b], cpu_device, warmups=WARMUPS, repeats=REPEATS)
    print(f"CPU: mean {cpu_mean:.4f}s  std {cpu_std:.4f}s  (runs: {cpu_times})")

    if gpu_device is not None:
        # Re-jit binding function explicitly to GPU device (so compilation target is GPU)
        heavy_fn_gpu = jax.jit(heavy_fn.lowerings()[0].fun, device=gpu_device) if False else heavy_fn  # fallback: same function works
        # Note: simply calling jax.jit(heavy_fn, device=gpu_device) may or may not change compilation target;
        # we rely on jax.device_put to place inputs on GPU and jax will compile for that device.
        print("\n--- TEST on GPU ---")
        gpu_mean, gpu_std, gpu_times = time_fn_on_device(heavy_fn, [a, b], gpu_device, warmups=WARMUPS, repeats=REPEATS)
        print(f"GPU: mean {gpu_mean:.4f}s  std {gpu_std:.4f}s  (runs: {gpu_times})")

        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
        print(f"\n--- RESULTS ---\nGPU is ~{speedup:.2f}x faster than CPU (means).")
    else:
        print("\nNo GPU available for comparison.")
    print()

if __name__ == "__main__":
    main()
