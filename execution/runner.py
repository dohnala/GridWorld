from collections import deque
from timeit import default_timer as timer
import logging

import numpy as np

from agents.agent import RunPhase


class TrainResult:
    def __init__(self, rolling_mean_window):
        self.num_episodes = 0
        self.rewards_per_episode = deque(maxlen=rolling_mean_window)
        self.steps_per_episode = deque(maxlen=rolling_mean_window)
        self.wins = deque(maxlen=rolling_mean_window)
        self.losses_per_episode = deque(maxlen=rolling_mean_window)
        self.time = 0

    def add_result(self, reward, steps, has_won, loss=None):
        self.num_episodes += 1
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(steps)
        self.wins.append(1 if has_won else 0)
        self.losses_per_episode.append(loss if loss else np.nan)

    def get_accuracy(self):
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        return np.mean(self.steps_per_episode)

    def get_mean_loss(self):
        return np.mean(self.losses_per_episode)


class EvalResult:
    def __init__(self):
        self.num_episodes = 0
        self.rewards_per_episode = deque()
        self.steps_per_episode = deque()
        self.wins = deque()
        self.time = 0

    def add_result(self, reward, steps, has_won):
        self.num_episodes += 1
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(steps)
        self.wins.append(1 if has_won else 0)

    def get_accuracy(self):
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        return np.mean(self.steps_per_episode)


class RunResult:
    def __init__(self, train_results, eval_results):
        self.train_episodes = sum(result.num_episodes for result in train_results)
        self.eval_episodes = eval_results[-1].num_episodes
        self.accuracy = eval_results[-1].get_accuracy()
        self.reward = eval_results[-1].get_mean_reward()
        self.steps = eval_results[-1].get_mean_steps()
        self.train_time = sum(result.time for result in train_results)
        self.eval_time = sum(result.time for result in eval_results)


class Runner:
    """
    Runner provides an abstraction for executing agent on given environment.
    """

    def __init__(self, env, agent):
        """
        Initialize runner.

        :param env: environment
        :param agent: agent
        """
        self.env = env
        self.agent = agent
        self.logger = logging.getLogger("root")

    def run(self, train_episodes, eval_episodes, eval_after, log_after, termination_cond=None):
        """
        Run agent for given number of episodes on the environment.

        :param train_episodes: number of training episodes
        :param eval_episodes: number of evaluating episodes
        :param eval_after: number of episodes before evaluation
        :param log_after: number of episodes before logging
        :param termination_cond: function which takes an evaluation result and decides if training should terminate
        :return: evaluation result
        """
        train_results = []
        eval_results = []

        current_episode = 0

        while True:
            remaining_episodes = train_episodes - current_episode
            num_episodes = eval_after if remaining_episodes > eval_after else remaining_episodes

            # Run training phase
            train_result = self.__train__(num_episodes, current_episode, train_episodes, log_after)

            # Store training result
            train_results.append(train_result)

            # Update episodes
            current_episode += num_episodes

            # Run evaluation phase
            eval_result = self.__eval__(eval_episodes)

            # Store evaluation result
            eval_results.append(eval_result)

            # If termination condition is defined and evaluated as True, break loop
            if termination_cond and termination_cond(eval_result):
                self.logger.info("Termination condition passed")
                self.logger.info("")
                break

            # If number of episodes exceed total number of training episodes, break loop
            if current_episode >= train_episodes:
                break

        # Create run result
        result = RunResult(train_results, eval_results)

        # Log run result
        self.__log_run_result__(result)
        self.logger.info("")

        return result

    def __train__(self, num_episodes, current_episode, total_episodes, log_after):
        """
        Run a training phase.

        :param num_episodes: number of episodes to train in this phase
        :param current_episode: episode before training phase
        :param total_episodes: total number of training episodes
        :param log_after: number of episodes before logging
        :return: result
        """
        start = timer()

        result = TrainResult(log_after)

        if num_episodes == 0:
            return result

        for episode in range(num_episodes):
            episode_reward = 0

            # Reset an environment before episode
            state = self.env.reset()

            while not self.env.is_terminal():
                # Get agent's action
                action = self.agent.act(state, RunPhase.TRAIN)

                # Execute given action in environment
                reward, next_state, done = self.env.step(action)

                # Pass observed transition to the agent
                self.agent.observe(state, action, reward, next_state, done)

                episode_reward += reward

                # Update state
                state = next_state

            # Add episode result
            result.add_result(
                reward=episode_reward,
                steps=self.env.state.step,
                has_won=self.env.has_won(),
                loss=self.agent.last_loss)

            # Log result
            if result.num_episodes == num_episodes or result.num_episodes % log_after == 0:
                self.__log_train_result__(result, current_episode, total_episodes)

        result.time = timer() - start

        self.logger.info("")

        return result

    def __eval__(self, num_episodes):
        """
        Run an evaluation phase.

        :param num_episodes: number of evaluation episodes
        :return: result
        """
        start = timer()

        result = EvalResult()

        if num_episodes == 0:
            return result

        for episode in range(num_episodes):
            episode_reward = 0

            # Reset an environment before episode
            state = self.env.reset()

            while not self.env.is_terminal():
                # Get agent's action
                action = self.agent.act(state, RunPhase.EVAL)

                # Execute given action in environment
                reward, next_state, done = self.env.step(action)

                episode_reward += reward

                # Update state
                state = next_state

            # Add episode result
            result.add_result(
                reward=episode_reward,
                steps=self.env.state.step,
                has_won=self.env.has_won())

        result.time = timer() - start

        # Log result
        self.__log_eval_result__(result)
        self.logger.info("")

        return result

    def __log_train_result__(self, result, current_episode, num_episodes):
        self.logger.info("Training {:4d}/{} - loss:{:>11f}, accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
            current_episode + result.num_episodes,
            num_episodes,
            result.get_mean_loss(),
            result.get_accuracy(),
            result.get_mean_reward(),
            result.get_mean_steps()))

    def __log_eval_result__(self, result):
        self.logger.info("Evaluation {:4d}/{} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
            result.num_episodes,
            result.num_episodes,
            result.get_accuracy(),
            result.get_mean_reward(),
            result.get_mean_steps()))

    def __log_run_result__(self, result):
        self.logger.info("Run result - train_episodes:{:6d}, eval_episodes:{:4d}, accuracy:{:7.2f}%, reward:{:6.2f}, "
                         "steps:{:6.2f}, train_time:{:7.2f}s".format(
            result.train_episodes,
            result.eval_episodes,
            result.accuracy,
            result.reward,
            result.steps,
            result.train_time))
