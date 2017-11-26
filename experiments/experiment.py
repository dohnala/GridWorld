import argparse
import logging.config
import os

from env.env import GridWorldEnv
from env.tasks import find_task

logging.config.fileConfig("logging.conf")


class Experiment:
    """
    Experiment in which agent tries to complete given task in environment.
    """

    def __init__(self, task_name):
        self.task_name = task_name
        self.task = find_task(task_name)
        self.parser = self.create_parser()
        self.logger = logging.getLogger("root")

    def create_agent(self, width, height, num_action):
        """
        Create agent.

        :param width: width
        :param height: height
        :param num_action: number of actions
        :return: agent
        """
        pass

    def create_runner(self, env_creator, agent_creator):
        """
        Create runner.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
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
        task = self.task

        def env_creator():
            return GridWorldEnv(task)

        def agent_creator():
            agent = self.create_agent(task.width, task.height, len(task.get_actions()))

            # Loading the agent state
            if args.load:
                if os.path.isfile(args.load):
                    agent.load(args.load)
                    self.logger.info("Agent loaded from {}".format(args.load))
                else:
                    self.logger.error("Agent couldn't be loaded. File {} doesn't exist".format(args.load))
                self.logger.info("")

            return agent

        def save(run, agent):
            # Saving the agent state
            if args.save:
                agent.save(args.save)
                self.logger.info("")
                self.logger.info("Agent saved to {}".format(args.save))

        runner = self.create_runner(env_creator, agent_creator)

        if runner is None:
            raise ValueError("No runner specified")

        self.logger.info("")

        if args.train:
            # Train
            runner.train(args.train, args.eval, args.eval_after, args.runs, self.termination_cond, save)
        elif args.eval:
            # Evaluate
            runner.eval(args.eval, args.runs)

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
        parser.add_argument('--runs', type=int, default=1, help='# of runs to average results')

        return parser
