from timeit import default_timer as timer

from agents.agent import RunPhase
from execution.result import EvalEpisodeResult, EvalResult, RunResult, TrainResult


class Runner:
    """
    Runner provides an abstraction for executing agent on given environment.
    """

    def __init__(self, env_fn, agent, seed=None):
        """
        Initialize runner.

        :param env_fn: function to create environment
        :param agent: agent
        :param seed: random seed
        """
        self.env_fn = env_fn
        self.agent = agent
        self.seed = seed

    def eval(self, eval_episodes):
        """
        Evaluate agent for given number of episodes.

        :param eval_episodes: number of evaluating episodes
        :return: result
        """
        eval_result = self.__eval_episodes__(self.env_fn(), self.agent, eval_episodes)

        return RunResult([], [eval_result])

    @staticmethod
    def __train_step__(env, agent):
        """
        Train given agent on given environment for one step.

        :param env: environment
        :param agent: agent
        :return: None
        """
        state = env.state

        # Get agent's action
        action = agent.act(state, RunPhase.TRAIN)

        # Execute given action in environment
        reward, next_state, done = env.step(action)

        # Pass observed transition to the agent
        agent.observe(state, action, reward, next_state, done)

        # Reset environment when episode ends
        if done:
            env.reset()

    def __train_steps__(self, env, agent, num_steps):
        """
        Train given agent on given environment for number of steps.

        :param env: environment
        :param agent: agent
        :param num_steps: number of steps
        :return: train result
        """
        start = timer()

        for step in range(num_steps):
            self.__train_step__(env, agent)

        return TrainResult(num_steps, timer() - start)

    @staticmethod
    def __eval_episode__(env, agent):
        """
        Eval agent on given environment for one episode.

        :param env: environment
        :param agent: agent
        :return: evaluation episode result
        """
        episode_reward = 0

        # Reset an environment before episode
        state = env.reset()

        while not env.is_terminal():
            # Get agent's action
            action = agent.act(state, RunPhase.EVAL)

            # Execute given action in environment
            reward, next_state, done = env.step(action)

            episode_reward += reward

            # Update state
            state = next_state

        # Return episode result
        return EvalEpisodeResult(
            reward=episode_reward,
            steps=env.state.step,
            has_won=env.has_won())

    def __eval_episodes__(self, env, agent, num_episodes):
        """
        Eval agent on given environment for given number of episodes.

        :param env: environment
        :param agent: agent
        :param num_episodes: number of episodes
        :return: eval result
        """
        result = EvalResult()

        start = timer()

        for episode in range(num_episodes):
            result.add_result(self.__eval_episode__(env, agent))

        result.time = timer() - start

        return result
