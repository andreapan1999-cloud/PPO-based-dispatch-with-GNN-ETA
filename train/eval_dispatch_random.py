import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sim.env_dispatch_ppo import DispatchPPOEnv

def make_env():
    return DispatchPPOEnv(
        data_dir="outputs",
        duration=180,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-2.0,
        wait_penalty=-0.1,
    )

def run_episode(env):
    obs = env.reset()
    done = [False]
    ep_reward = 0.0
    while not done[0]:
        a = np.array([env.action_space.sample()])
        obs, reward, done, info = env.step(a)
        ep_reward += float(reward[0])
    return ep_reward

if __name__ == "__main__":
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load("outputs/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    rs = [run_episode(venv) for _ in range(20)]
    rs = np.array(rs, dtype=np.float64)
    print(f"reward mean/std: {rs.mean():.3f} / {rs.std():.3f}")