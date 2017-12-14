class VecEnv:
    """
    Vectorized environment.
    """

    def get_states(self):
        """
        Return current states.

        :return: list of current states
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

        :return: list of states
        """
        pass

    def close(self):
        """
        Close this vectorized environment.

        :return: None
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

    def get_states(self):
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

        return list(rewards), list(next_states), list(dones)

    def reset(self):
        return [env.reset() for env in self.envs]
