from typing import Dict
import gymnasium as gym
import numpy as np
import time
from tqdm.auto import trange


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens = [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}


def evaluate_toy(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens= [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}
