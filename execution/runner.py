from timeit import default_timer as timer
import logging

import numpy as np

from agents.agent import RunPhase
from execution.result import AverageRunResult, RunResult, TrainResult, EvalResult


class Runner:
    """
    Runner provides an abstraction for executing agent on given environment.
    """

    def __init__(self, env_creator, agent_creator):
        """
        Initialize runner.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
        """
        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.logger = logging.getLogger("root")

    def run(self, train_episodes, eval_episodes, eval_after, runs=1, termination_cond=None, after_run=None):
        """
        Run agent for given number of episodes on the environment several times.

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
            result = self.__run__(i, train_episodes, eval_episodes, eval_after, termination_cond, after_run)

            # Store run result
            run_results.append(result)

        # Create average run result
        result = AverageRunResult(run_results)

        # Log run result
        self.__log_average_run_result__(result)

        return result

    def __run__(self, run, train_episodes, eval_episodes, eval_after, termination_cond=None, after_run=None):
        """
        Run agent for given number of episodes on the environment.

        :param run: current run
        :param train_episodes: number of training episodes
        :param eval_episodes: number of evaluating episodes
        :param eval_after: number of episodes before evaluation
        :param termination_cond: function which takes an evaluation result and decides if training should terminate
        :param after_run: function called after run which takes an agent
        :return: evaluation result
        """

        # Create env and agent for the run
        env = self.env_creator()
        agent = self.agent_creator()

        train_results = []
        eval_results = []

        current_episode = 0

        while True:
            remaining_episodes = train_episodes - current_episode
            num_episodes = eval_after if remaining_episodes > eval_after else remaining_episodes

            # Run training phase
            train_result = self.__train__(env, agent, num_episodes)

            # Store training result
            train_results.append(train_result)

            # Update episodes
            current_episode += num_episodes

            # Run evaluation phase
            eval_result = self.__eval__(env, agent, eval_episodes)

            # Log evaluation result
            self.__log_eval_result__(current_episode, eval_result)

            # Store evaluation result
            eval_results.append(eval_result)

            # If termination condition is defined and evaluated as True, break loop
            if termination_cond and termination_cond(eval_result):
                self.logger.info("")
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
        self.logger.info("-" * 150)

        # Call after run callback
        if after_run:
            after_run(run, agent)

        return result

    def __train__(self, env, agent, num_episodes):
        """
        Run a training phase.

        :param env: environment
        :param agent: agent
        :param num_episodes: number of episodes to train in this phase
        :return: result
        """
        start = timer()

        result = TrainResult(num_episodes)

        if num_episodes == 0:
            return result

        for episode in range(num_episodes):
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

            # Add episode result
            result.add_result(
                reward=episode_reward,
                steps=env.state.step,
                has_won=env.has_won(),
                loss=agent.last_loss)

        result.time = timer() - start

        return result

    def __eval__(self, env, agent, num_episodes):
        """
        Run an evaluation phase.

        :param env: environment
        :param agent: agent
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
            state = env.reset()

            while not env.is_terminal():
                # Get agent's action
                action = agent.act(state, RunPhase.EVAL)

                # Execute given action in environment
                reward, next_state, done = env.step(action)

                episode_reward += reward

                # Update state
                state = next_state

            # Add episode result
            result.add_result(
                reward=episode_reward,
                steps=env.state.step,
                has_won=env.has_won())

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
