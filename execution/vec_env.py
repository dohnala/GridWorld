class VecEnv:
    """
    Vectorized environment.
    """

    def get_state(self):
        """
        Return current state.

        :return: current state
        """
        pass

    def step(self, actions):
        """
        Perform one step with given actions return (reward, next_state, done) tuples.

        :param actions: list of actions to perform
        :return: list of (reward, next_state, done) tuples
        """
        pass

    def reset(self):
        """
        Reset all environments.
        """
        pass

    def close(self):
        """
        Close this vectorized environment.
        :return:
        """
        pass


class SyncVecEnv(VecEnv):
    """
    Synchronous implementation of vectorized environment.
    """

    def __init__(self, env_fn, num_env):
        """
        Initialize vectorized environment.

        :param env_fn: function to create environment
        :param num_env: number of environments.
        """
        self.envs = [env_fn() for _ in range(num_env)]

    def get_state(self):
        return [env.state for env in self.envs]

    def step(self, actions):
        # Perform actions
        results = [env.step(action) for (action, env) in zip(actions, self.envs)]

        # Convert results
        rewards, next_states, dones = zip(*results)

        # Reset environments if done flag is set
        for (i, done) in enumerate(dones):
            if done:
                self.envs[i].reset()

        return rewards, next_states, dones

    def reset(self):
        for env in self.envs:
            env.reset()
