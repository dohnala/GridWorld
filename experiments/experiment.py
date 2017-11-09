import argparse
import logging.config
import os
from collections import deque
from timeit import default_timer as timer

import numpy as np

from env.env import GridWorldEnv

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("root")


class Result:
    """
    Store training/evaluation result across multiple number of episodes.
    """

    def __init__(self, rolling_mean_window):
        self.start = timer()
        self.num_episodes = 0
        self.rewards_per_episode = deque(maxlen=rolling_mean_window)
        self.steps_per_episode = deque(maxlen=rolling_mean_window)
        self.wins = deque(maxlen=rolling_mean_window)
        self.losses_per_episode = deque(maxlen=rolling_mean_window)

        self.observers = []

    def add_result(self, episode_result):
        """
        Add an episode result and notify observers.

        :param episode_result: result of an episode
        """
        self.num_episodes += 1
        self.rewards_per_episode.append(episode_result.reward)
        self.steps_per_episode.append(episode_result.steps)
        self.wins.append(1 if episode_result.has_won else 0)
        self.losses_per_episode.append(episode_result.loss if episode_result.loss else np.nan)

        # Notify observers
        for observer in self.observers:
            observer(self)

    def get_accuracy(self):
        """
        Return win accuracy in percentages.

        :return: win accuracy in percentages
        """
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        """
        Return mean of rewards gained for each episode.

        :return: mean of rewards gained for each episode
        """
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        """
        Return mean of steps done for each episode.

        :return: mean of steps done for each episode
        """
        return np.mean(self.steps_per_episode)

    def get_mean_loss(self):
        """
        Return mean of losses.

        :return: mean of losses
        """
        return np.mean(self.losses_per_episode)

    def add_observer(self, observer):
        """
        Add observer which will be notified when result is changed.

        :param observer: observer
        """
        self.observers.append(observer)


class Experiment:
    """
    Experiment in which agent tries to complete given task in environment.
    """

    def __init__(self, task):
        self.task = task
        self.parser = self.create_parser()

    def create_agent(self, env):
        """
        Create an agent.

        :param env: environment
        :return: agent
        """
        pass

    def run(self):
        """
        Run an experiment.
        """
        args = self.parser.parse_args()

        env = GridWorldEnv(self.task)
        agent = self.create_agent(env)

        if agent is None:
            raise ValueError("No agent specified")

        # Info
        logger.info("")
        logger.info("Task: {}".format(self.task))
        logger.info("Agent: {}".format(agent.name))
        logger.info("")

        # Loading the model
        if args.load:
            if os.path.isfile(args.load):
                agent.load(args.load)
                logger.info("Model loaded from {}".format(args.load))
            else:
                logger.error("Model couldn't be loaded. File {} doesn't exist".format(args.load))
            logger.info("")

        # Training
        if args.train:
            # noinspection PyTypeChecker
            self.train(agent, args.train, args.log_every)

            # Saving the model
            if args.save:
                agent.save(args.save)
                logger.info("Model saved to {}".format(args.save))
                logger.info("")

        # Evaluating
        if args.eval:
            # noinspection PyTypeChecker
            self.eval(agent, args.eval, args.log_every)

    @staticmethod
    def train(agent, num_episodes, log_every):
        """
        Train an agent for given number of episodes.

        :param agent: agent
        :param num_episodes: number of episodes to train
        :param log_every: log result every given number of episodes
        """

        def handle_train_result(result):
            if result == num_episodes or result.num_episodes % log_every == 0:
                logger.info("Episode {:4d}/{} - loss:{:>11f}, accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
                    result.num_episodes,
                    num_episodes,
                    result.get_mean_loss(),
                    result.get_accuracy(),
                    result.get_mean_reward(),
                    result.get_mean_steps()))

        logger.info("Training started")
        logger.info("")

        train_result = Result(log_every)
        train_result.add_observer(handle_train_result)
        agent.train(num_episodes, train_result)

        logger.info("")
        logger.info("Training finished after {:.2f}s".format(timer() - train_result.start))
        logger.info("")

    @staticmethod
    def eval(agent, num_episodes, log_every):
        """
        Evaluate an agent for given number of episodes.

        :param agent: agent
        :param num_episodes: number of episodes to evaluate
        :param log_every: log result every given number of episodes
        """

        def handle_eval_result(result):
            if result == num_episodes or result.num_episodes % log_every == 0:
                logger.info("Episode {:4d}/{} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
                    result.num_episodes,
                    num_episodes,
                    result.get_accuracy(),
                    result.get_mean_reward(),
                    result.get_mean_steps()))

        logger.info("Evaluating started")
        logger.info("")

        eval_result = Result(num_episodes)
        eval_result.add_observer(handle_eval_result)
        agent.eval(num_episodes, eval_result)

        logger.info("")
        logger.info("Evaluating finished after {:.2f}s".format(timer() - eval_result.start))
        logger.info("")

    @staticmethod
    def create_parser():
        """
        Create command line argument parser.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Grid world experiment')

        parser.add_argument('--train', type=int, help='# of train episodes (default: 100)')
        parser.add_argument('--eval', type=int, help='# of eval episodes (default: 10)')
        parser.add_argument('--save', type=str, help="file to save model to")
        parser.add_argument('--load', type=str, help="file to load model from")
        parser.add_argument('--log_every', type=int, default=10, help='log every # of episodes (default: 10)')

        return parser
