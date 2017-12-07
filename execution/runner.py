import random
from timeit import default_timer as timer

import numpy as np
import torch

from agents.agent import RunPhase
from execution.result import TrainEpisodeResult, EvalEpisodeResult, EvalResult, TrainResult, RunResult


class Runner:
    """
    Runner provides an abstraction for executing agent on given environment.
    """

    def __init__(self, env, agent, seed=1):
        """
        Initialize runner.

        :param env: environment
        :param agent: agent
        :param seed: random seed
        """
        self.env = env
        self.agent = agent
        self.seed = seed

        if seed:
            self.__set_seed__(self.seed)

    def eval(self, eval_episodes):
        """
        Evaluate agent for given number of episodes.

        :param eval_episodes: number of evaluating episodes
        :return: result
        """
        eval_result = self.__eval_episodes__(self.env, self.agent, eval_episodes)

        return RunResult([], [eval_result])

    @staticmethod
    def __train_episode__(env, agent):
        """
        Train given agent on given environment for one episode.

        :param env: environment
        :param agent: agent
        :return: train episode result
        """
        episode_reward = 0

        # Reset an environment before episode
        state = env.reset()

        while not env.is_terminal():
            # Get agent's action
            action = agent.act(state, RunPhase.TRAIN)

            # Execute given action in environment
            reward, next_state, done = env.step(action)

            # Pass observed transition to the agent
            agent.observe(state, action, reward, next_state, done)

            episode_reward += reward

            # Update state
            state = next_state

        # Return episode result
        return TrainEpisodeResult(
            reward=episode_reward,
            steps=env.state.step,
            has_won=env.has_won(),
            loss=agent.last_loss)

    def __train_episodes__(self, env, agent, num_episodes):
        """
        Train given agent on given environment for one episode.

        :param env: environment
        :param agent: agent
        :param num_episodes: number of episodes
        :return: train result
        """
        start = timer()

        result = TrainResult(num_episodes)

        for episode in range(num_episodes):
            result.add_result(self.__train_episode__(env, agent))

        result.time = timer() - start

        return result

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

    @staticmethod
    def __set_seed__(seed):
        """
        Set random seed.

        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
