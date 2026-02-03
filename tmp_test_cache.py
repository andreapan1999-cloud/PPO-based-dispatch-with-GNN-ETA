from sim.env_dispatch import DispatchEnv

print("first", flush=True)

env1 = DispatchEnv(
    "outputs",
    180,
    use_gnn_eta=True,
    calibrate_gnn=True,
    calib_samples=2000,
    verbose=True,
)

print("\nsecond", flush=True)

env2 = DispatchEnv(
    "outputs",
    180,
    use_gnn_eta=True,
    calibrate_gnn=True,
    calib_samples=2000,
    verbose=True,
)