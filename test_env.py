from sim.env_dispatch import DispatchEnv

env = DispatchEnv(data_dir="outputs", duration=180)

obs, info = env.reset()
print("reset obs:", obs)

done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # random policy
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print("episode finished")
print("total reward:", total_reward)
