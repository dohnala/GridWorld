import logging
from timeit import default_timer as timer

import numpy as np
import torch
import random
from agents.agent import RunPhase
from execution.result import AverageRunResult, TrainEpisodeResult, EvalEpisodeResult, EvalResult, TrainResult, RunResult


class Runner:
    """
    Runner provides an abstraction for executing agent on given environment.
    """

    def __init__(self, env_creator, agent_creator, seed=1):
        """
        Initialize runner.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
        :param seed: random seed
        """
        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.seed = seed
        self.logger = logging.getLogger("root")

        if seed:
            self.__set_seed__(self.seed)

    def train(self, train_episodes, eval_episodes, eval_after, runs=1, termination_cond=None, after_run=None):
        """
        Train agent for given number of episodes on the environment several times.

        :param train_episodes: number of training episodes
        :param eval_episodes: number of evaluating episodes
        :param eval_after: number of episodes before evaluation
        :param runs: number of runs to average results
        :param termination_cond: function which takes an evaluation result and decides if training should terminate
        :param after_run: function called after run which takes an agent
        :return: average result across runs
        """
        run_results = []

        for i in range(runs):
            self.logger.info("# Run {}/{}".format(i + 1, runs))
            self.logger.info("")

            # Run agent on the environment
            result = self.__train__(i, train_episodes, eval_episodes, eval_after, termination_cond, after_run)

            # Store run result
            run_results.append(result)

        # Create average run result
        result = AverageRunResult(run_results)

        # Log run result
        self.__log_average_run_result__(result)

        return result

    def eval(self, eval_episodes, runs=1):
        """
        Evaluate agent for given number of episodes several times.

        :param eval_episodes: number of evaluating episodes
        :param runs: number of runs to average results
        :return: average result across runs
        """
        run_results = []

        for i in range(runs):
            self.logger.info("# Run {}/{}".format(i + 1, runs))
            self.logger.info("")

            env = self.env_creator()
            agent = self.agent_creator()

            # Evaluate agent
            eval_result = self.__eval_episodes__(env, agent, eval_episodes)

            run_result = RunResult([], [eval_result])

            self.__log_run_result__(run_result)
            self.logger.info("-" * 150)

            # Store run result
            run_results.append(run_result)

        # Create average run result
        result = AverageRunResult(run_results)

        # Log run result
        self.__log_average_run_result__(result)

        return result

    def __train__(self, run, train_episodes, eval_episodes, eval_after, termination_cond=None, after_run=None):
        """
        Train agent for given number of episodes on the environment.

        :param run: current run
        :param train_episodes: number of training episodes
        :param eval_episodes: number of evaluating episodes
        :param eval_after: number of episodes before evaluation
        :param termination_cond: function which takes an evaluation result and decides if training should terminate
        :param after_run: function called after run which takes an agent
        :return: evaluation result
        """
        pass

    @staticmethod
    def __train_episode__(env, agent):
        """
        Train given agent on given environment for one episode.

        :param env: environment
        :param agent: agent
        :return: episode result
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
        :return: episode result
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

    def __log_train_result__(self, result, current_episode, num_episodes):
        self.logger.info("Training {:4d}/{} - loss:{:>11f}, accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
            current_episode + result.num_episodes,
            num_episodes,
            result.get_mean_loss(),
            result.get_accuracy(),
            result.get_mean_reward(),
            result.get_mean_steps()))

    def __log_eval_result__(self, current_episode, result):
        self.logger.info("Evaluation at {:4d} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
            current_episode,
            result.get_accuracy(),
            result.get_mean_reward(),
            result.get_mean_steps()))

    def __log_run_result__(self, result):
        self.logger.info("Result - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}, train_time:{:5.2f}s".format(
            result.accuracy,
            result.reward,
            result.steps,
            result.train_time))

    def __log_average_run_result__(self, result):
        self.logger.info("# Run results")
        self.logger.info("")

        for i in range(result.num_runs):
            run_result = result.run_results[i]

            self.logger.info("Run {:2d} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}, train_time:{:5.2f}s".format(
                i + 1,
                run_result.accuracy,
                run_result.reward,
                run_result.steps,
                run_result.train_time))

        self.logger.info("")
        self.logger.info("# Average statistics")
        self.logger.info("")

        self.logger.info("Runs       - {}".format(result.num_runs))
        self.logger.info("Eval       - {} episodes".format(result.eval_episodes))

        self.logger.info("Accuracy   - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
            np.mean(result.accuracy_per_run),
            np.min(result.accuracy_per_run),
            np.max(result.accuracy_per_run),
            np.var(result.accuracy_per_run)))

        self.logger.info("Reward     - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
            np.mean(result.reward_per_run),
            np.min(result.reward_per_run),
            np.max(result.reward_per_run),
            np.var(result.reward_per_run)))

        self.logger.info("Steps      - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
            np.mean(result.steps_per_run),
            np.min(result.steps_per_run),
            np.max(result.steps_per_run),
            np.var(result.steps_per_run)))

        self.logger.info("Train time - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
            np.mean(result.train_time_per_run),
            np.min(result.train_time_per_run),
            np.max(result.train_time_per_run),
            np.var(result.train_time_per_run)))

        self.logger.info("")

    @staticmethod
    def __set_seed__(seed):
        """
        Set random seed.

        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
