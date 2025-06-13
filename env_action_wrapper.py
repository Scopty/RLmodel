import gymnasium as gym
import numpy as np

class ActionTupleWrapper(gym.Wrapper):
    """
    Converts flat actions (from PPO) to Tuple(action_type, stoploss_distance) for TradingEnv with stoploss enabled.
    """
    def __init__(self, env):
        super().__init__(env)
        self.is_tuple = hasattr(env.action_space, 'spaces') and len(env.action_space.spaces) == 2
        if self.is_tuple:
            # Flatten action space for PPO: Discrete(3) + Box(1,)
            from gymnasium.spaces import Box
            low = np.array([0, env.stoploss_min], dtype=np.float32)
            high = np.array([2, env.stoploss_max], dtype=np.float32)
            self.action_space = Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = env.action_space

    def step(self, action):
        if self.is_tuple:
            # PPO outputs [discrete, continuous], map to (int, float)
            action_type = int(np.round(np.clip(action[0], 0, 2)))
            stoploss_distance = float(np.clip(action[1], self.env.stoploss_min, self.env.stoploss_max))
            return self.env.step((action_type, stoploss_distance))
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
