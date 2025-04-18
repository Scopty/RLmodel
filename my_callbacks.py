class RewardCallback(BaseCallback):
    def __init__(self, verbose=0, debug=False):
        super(RewardCallback, self).__init__(verbose)
        self.debug = debug
        
        # Metrics for tracking actions and rewards
        self.episode_rewards = []
        self.episode_steps = []
        self.iteration_rewards = []
        self.iteration_invalid_actions = []
        self.invalid_actions = []
        self.valid_actions = []
        self.current_episode_steps = 0
        
        # Metrics for TensorBoard logging
        self.total_reward = 0
        self.reward = 0
        self.num_trades = 0

    def _on_step(self) -> bool:
        # Collect rewards and actions
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        
        if len(rewards) > 0:  # Check if rewards is not empty
            self.episode_rewards.extend(rewards)
        if len(actions) > 0:  # Check if actions is not empty
            infos = self.locals.get("infos", [])
            for idx, info in enumerate(infos):
                valid_actions = info.get("valid_actions", [0, 1, 2])
                action = actions[idx]
                if action not in valid_actions:
                    self.invalid_actions.append(action)
                else:
                    self.valid_actions.append(action)

        self.current_episode_steps += 1

        # Access the environment metrics using get_attr for SubprocVecEnv
        if isinstance(self.training_env, SubprocVecEnv):
            try:
                inner_envs = self.training_env.get_attr('env')  # ActionMasker
                for env in inner_envs:
                    if hasattr(env, 'env'):  # Unwrap ActionMasker
                        env = env.env
                    self.total_reward += getattr(env, "total_reward", 0)
                    self.num_trades += getattr(env, "round_trip_trades", 0)
            except Exception as e:
                if self.debug:
                    print(f"Failed to access env attributes: {e}")
        else:
            # For DummyVecEnv or single environments
            for env in self.training_env.envs:
                if hasattr(env, 'env'):
                    env = env.env
                self.total_reward += getattr(env, "total_reward", 0)
                self.num_trades += getattr(env, "round_trip_trades", 0)

        # TensorBoard logging
        self.logger.record("custom/num_trades", self.num_trades)
        self.logger.record("custom/total_reward", self.total_reward)

        # Entropy logging
        if hasattr(self.model.policy, "action_dist"):
            action_dist = self.model.policy.action_dist
            entropy = action_dist.entropy().mean().item()
            self.logger.record("policy/entropy", entropy)
        elif hasattr(self.model.policy, "get_distribution"):
            obs = self.locals.get("obs", [])
            if len(obs) > 0:  # Check if observations exist
                action_dist = self.model.policy.get_distribution(obs)
                entropy = action_dist.entropy().mean().item()
                self.logger.record("policy/entropy", entropy)

        # Value loss logging
        if "value_loss" in self.locals:
            value_loss = self.locals["value_loss"]
            self.logger.record("loss/value_loss", value_loss)

        # Episode done handling
        dones = self.locals.get("dones", [])
        if any(dones):
            self.episode_steps.append(self.current_episode_steps)
            self.current_episode_steps = 0
            total_reward = np.sum(self.episode_rewards)
            self.iteration_rewards.append(total_reward)
            self.episode_rewards = []

            invalid_count = len(self.invalid_actions)
            valid_count = len(self.valid_actions)
            self.iteration_invalid_actions.append(invalid_count)

            if self.debug:
                print(f"Invalid actions in this episode: {invalid_count}")
                print(f"Valid actions in this episode: {valid_count}")
                print(f"Invalid actions: {self.invalid_actions}")

            self.invalid_actions = []

        return True


# Initialize the callback
reward_callback = RewardCallback(debug=True)
