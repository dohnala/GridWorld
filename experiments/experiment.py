import argparse
import os
import random

import numpy as np
import torch

from env.env import GridWorldEnv
from execution.average_runner import AverageRunner
from execution.result import log_average_run_result
from utils.logging import logger


class Experiment:
    """
    Experiment in which agent tries to complete given task in environment.
    """

    def __init__(self):
        self.parser = self.create_parser()

    def define_task(self):
        """
        Define task which should be learned.

        :return: task
        """
        pass

    def define_agent(self, width, height, num_actions):
        """
        Define agent which should learn task.

        :param width: width
        :param height: height
        :param num_actions: number of actions
        :return: agent
        """
        pass

    def define_goal(self, result):
        """
        Define goal which agent should reach on given task.

        :param result: evaluation result
        :return: True if training process should be terminated
        """
        return False

    def train(self, env, agent, seed):
        """
        Train given agent on given environment and return result.

        :param env: environment
        :param agent: agent
        :param seed: random seed
        :return: result
        """
        pass

    def eval(self, env, agent, seed):
        """
        Evaluate given agent on given environment and return result.

        :param env: environment
        :param agent: agent
        :param seed: random seed
        :return: result
        """
        pass

    def run(self):
        """
        Run an experiment.
        """

        args = self.parser.parse_args()

        # Set random seed
        self.set_seed(args.seed)

        def run_op(op):
            # Create task
            task = self.define_task()

            # Create environment
            env = GridWorldEnv(task, seed=args.seed)

            # Create agent
            agent = self.define_agent(task.width, task.height, len(task.get_actions()))

            # Log experiment info
            self.log_info(task, agent)

            # Loading the agent state
            if args.load:
                if os.path.isfile(args.load):
                    agent.load(args.load)
                    logger.info("Agent loaded from {}".format(args.load))
                else:
                    logger.error("Agent couldn't be loaded. File {} doesn't exist".format(args.load))
                logger.info("")

            # Run op and return its result
            return op(env, agent)

        def run_train():
            def train_op(env, agent):
                # Train agent on environment
                result = self.train(env, agent, args.seed)

                # Saving the agent state
                if args.save:
                    agent.save(args.save)
                    logger.info("Agent saved to {}".format(args.save))
                    logger.info("")

                return result

            # Run train op and return its result
            return run_op(train_op)

        def run_eval():
            def eval_op(env, agent):
                # Evaluate agent on environment
                return self.eval(env, agent, args.seed)

            # Run eval op and return its result
            return run_op(eval_op)

        if args.train:
            # Train agent
            avg_result = AverageRunner(run_train).run(args.runs)

            log_average_run_result(avg_result)
        elif args.eval:
            # Evaluate agent
            avg_result = AverageRunner(run_eval).run(args.runs)

            log_average_run_result(avg_result)

    @staticmethod
    def set_seed(seed):
        """
        Set random seed.

        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def log_info(task, agent):
        logger.info("Task: {}".format(str(task)))
        logger.info("Agent: {}".format(str(agent.name)))
        logger.info("")

    @staticmethod
    def create_parser():
        """
        Create command line argument parser.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Grid world experiment')

        parser.add_argument('--train', action='store_true', help='train agent')
        parser.add_argument('--eval', action='store_true', help='evaluate agent')
        parser.add_argument('--save', type=str, help="file to save model to")
        parser.add_argument('--load', type=str, help="file to load model from")
        parser.add_argument('--runs', type=int, default=1, help='# of runs to average results')
        parser.add_argument('--seed', type=int, default=1, help='random seed')

        return parser
