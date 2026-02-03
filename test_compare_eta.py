from sim.env_dispatch import DispatchEnv

def run(env):
    env.reset()
    total = 0.0
    for _ in range(200):
        _, r, done, _, _ = env.step(1)
        total += r
        if done:
            break
    return total

baseline = DispatchEnv("outputs", 180, use_gnn_eta=False)
print("baseline total", run(baseline))

gnn = DispatchEnv("outputs", 180, use_gnn_eta=True, gnn_ckpt="outputs/eta_gnn.pt")
print("gnn total", run(gnn))
