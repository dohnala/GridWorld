import argparse
import logging.config
import os
from timeit import default_timer as timer

from env.env import GridWorldEnv

logging.config.fileConfig("logging.conf")


class Experiment:
    """
    Experiment in which agent tries to complete given task in environment.
    """

    def __init__(self, task):
        self.task = task
        self.parser = self.create_parser()
        self.logger = logging.getLogger("root")

    def create_runner(self, env):
        """
        Create runner.

        :param env: environment
        :return: runner
        """
        pass

    def termination_cond(self, result):
        """
        Define termination condition to terminate training process after some evaluation result is
        achieved.

        :param result: evaluation result
        :return: True if training process should be terminated
        """
        return False

    def run(self):
        """
        Run an experiment.
        """
        args = self.parser.parse_args()

        env = GridWorldEnv.for_task_name(self.task)
        runner = self.create_runner(env)

        if runner is None:
            raise ValueError("No runner specified")

        # Info
        self.logger.info("")
        self.logger.info("Task: {}".format(self.task))
        self.logger.info("Agent: {}".format(runner.agent.name))
        self.logger.info("")

        # Loading the model
        if args.load:
            if os.path.isfile(args.load):
                runner.agent.load(args.load)
                self.logger.info("Model loaded from {}".format(args.load))
            else:
                self.logger.error("Model couldn't be loaded. File {} doesn't exist".format(args.load))
            self.logger.info("")

        self.logger.info("Experiment started")
        self.logger.info("")

        start = timer()

        runner.run(args.train, args.eval, args.eval_after, args.log_after, self.termination_cond)

        self.logger.info("Experiment finished after {:.2f}s".format(timer() - start))
        self.logger.info("")

        # Saving the model
        if args.save:
            runner.agent.save(args.save)
            self.logger.info("Model saved to {}".format(args.save))
            self.logger.info("")

    @staticmethod
    def create_parser():
        """
        Create command line argument parser.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Grid world experiment')

        parser.add_argument('--train', type=int, default=0, help='# of train episodes')
        parser.add_argument('--eval', type=int, default=0, help='# of eval episodes')
        parser.add_argument('--eval_after', type=int, default=100, help='evaluate after # of episodes')
        parser.add_argument('--save', type=str, help="file to save model to")
        parser.add_argument('--load', type=str, help="file to load model from")
        parser.add_argument('--log_after', type=int, default=10, help='log result after # of episodes')

        return parser
