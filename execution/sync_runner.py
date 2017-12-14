import os
from timeit import default_timer as timer

from agents.agent import RunPhase
from execution import Runner
from execution.result import RunResult, log_eval_result, TrainResult
from execution.vec_env import AsyncVecEnv, SyncVecEnv
from utils.logging import logger


class SyncRunner(Runner):
    """
    Synchronous runner implementation.
    """

    def __init__(self, env_fn, agent, num_processes=1, seed=None):
        """
        Initialize runner.

        :param env_fn: function to create environment
        :param agent: agent
        :param num_processes: number of processes
        :param seed: random seed
        """
        super(SyncRunner, self).__init__(env_fn, agent, seed)

        self.num_processes = num_processes

    def train(self, train_steps, eval_every_steps, eval_episodes, goal=None):
        """
        Train agent for given number of steps.

        :param train_steps: number steps to train agent
        :param eval_every_steps: evaluate agent every `eval_every_steps` steps
        :param eval_episodes: number of episode to evaluate agent for
        :param goal: goal which can terminate training if it is reached
        :return: result
        """
        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Create environments
        train_envs = SyncVecEnv(self.env_fn, self.num_processes, self.seed)
        eval_env = self.env_fn()

        # results
        train_results = []
        eval_results = []

        current_step = 0

        while True:
            remaining_steps = train_steps - current_step
            num_steps = eval_every_steps if remaining_steps > eval_every_steps else remaining_steps

            # Train agent
            train_result = self.__train_steps__(train_envs, self.agent, num_steps)

            # Store training result
            train_results.append(train_result)

            # Update steps
            current_step += num_steps

            # Evaluate agent
            eval_result = self.__eval_episodes__(eval_env, self.agent, eval_episodes)

            # Log evaluation result
            log_eval_result(current_step, eval_result)

            # Store evaluation result
            eval_results.append(eval_result)

            # If goal is defined and evaluated as True, break loop
            if goal and goal(eval_result):
                logger.info("")
                logger.info("Termination condition passed")
                logger.info("")
                break

            # If number of steps exceed total number of training steps, break loop
            if current_step >= train_steps:
                logger.info("")
                break

        # Close vectorized environments
        train_envs.close()

        # Return run result
        return RunResult(train_results, eval_results)

    def __train_steps__(self, envs, agent, num_steps):
        """
        Train given agent on given environments for number of steps.

        :param envs: environments
        :param agent: agent
        :param num_steps: number of steps
        :return: train result
        """
        start = timer()

        for step in range(int(num_steps / self.num_processes)):
            # Get current states
            states = envs.get_states()

            # Get agent's action
            actions = agent.act(states, RunPhase.TRAIN)

            # Execute given action in environment
            rewards, next_states, dones = envs.step(actions)

            # Pass observed transition to the agent
            agent.observe(states, actions, rewards, next_states, dones)

        return TrainResult(num_steps, timer() - start)
