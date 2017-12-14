import copy
import os
import time
from timeit import default_timer as timer

import torch.multiprocessing as mp

from agents.agent import RunPhase
from execution import Runner
from execution.result import RunResult, log_eval_result, TrainResult, EvalResult, EvalEpisodeResult
from utils.logging import logger
from utils.multiprocessing import deserialize, serialize
from utils.seed import set_seed


class AsyncRunner(Runner):
    """
    Asynchronous runner implementation which runs multiple workers in separate process to train agent.
    """

    def __init__(self, env_fn, agent, num_processes, seed=None):
        """
        Initialize runner.

        :param env_fn: function to create environment
        :param agent: agent
        :param num_processes: number of processes
        :param seed: random seed
        """
        super(AsyncRunner, self).__init__(env_fn, agent, seed)

        self.num_processes = num_processes

    def train(self, train_steps, eval_every_sec, eval_episodes, goal=None):
        """
        Train agent for given number of steps.

        :param train_steps: number of steps to train agent
        :param eval_every_sec: evaluate agent every `eval_every_sec` seconds
        :param eval_episodes: number of episode to evaluate agent for
        :param goal: goal which can terminate training if it is reached
        :return: result
        """
        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Flag indicating that training is finished
        stop_flag = mp.Event()

        # Number of steps for each worker
        workers_train_steps = int(train_steps / self.num_processes)

        # Workers' current steps
        workers_steps = mp.Array('i', self.num_processes)

        # Queue where the final result is put
        result_queue = mp.Queue()

        processes = []

        start = timer()

        # Create and start evaluation process
        eval_process = EvalProcess(
            env_fn_serialized=serialize(self.env_fn),
            agent=self.agent,
            seed=self.seed + self.num_processes if self.seed is not None else None,
            train_steps=train_steps,
            eval_every_sec=eval_every_sec,
            eval_episodes=eval_episodes,
            goal_serialized=serialize(goal),
            stop_flag=stop_flag,
            workers_steps=workers_steps,
            result_queue=result_queue)

        eval_process.start()
        processes.append(eval_process)

        # Create and start worker processes
        for worker in self.agent.create_workers(self.num_processes):
            worker_process = WorkerProcess(
                env_fn_serialized=serialize(self.env_fn),
                worker=worker,
                seed=self.seed + worker.worker_id if self.seed is not None else None,
                train_steps=workers_train_steps,
                workers_steps=workers_steps,
                stop_flag=stop_flag)

            worker_process.start()
            processes.append(worker_process)

        # Wait until all processes finish execution
        [process.join() for process in processes]

        # Get result from queue
        result = result_queue.get()
        result.train_time = timer() - start

        return result


class WorkerProcess(mp.Process):
    """
    Process which trains worker on isolated environment.
    """

    def __init__(self, env_fn_serialized, worker, seed, train_steps, workers_steps, stop_flag, batch_steps=10):
        """
        Initialize worker process.

        :param env_fn_serialized: serialized function to create environment
        :param worker: worker
        :param seed: seed
        :param train_steps: number of training steps
        :param workers_steps: array for storing workers' current step
        :param stop_flag: flag indicating that training should be terminated
        :param batch_steps: how many steps should be trained before checking conditions
        """
        super(WorkerProcess, self).__init__()

        self.env = deserialize(env_fn_serialized)()
        self.worker = worker
        self.seed = seed
        self.train_steps = train_steps
        self.workers_steps = workers_steps
        self.stop_flag = stop_flag
        self.batch_steps = batch_steps

    def run(self):
        """
        Run process.

        :return: None
        """
        # Set random seed for this process
        set_seed(self.seed)

        # Initialize worker's current step
        self.workers_steps[self.worker.worker_id] = 0

        # Train until stop flag is set or number of training steps is reached
        while not self.stop_flag.is_set() and self.workers_steps[self.worker.worker_id] < self.train_steps:
            # Train worker for batch steps
            self.__train__(self.batch_steps)

    def __train__(self, num_steps):
        """
        Train worker for given number of steps.

        :param num_steps: number of steps
        :return: None
        """
        start = timer()

        for step in range(num_steps):
            # Get current state
            state = self.env.state

            # Get worker's action
            action = self.worker.act([state], RunPhase.TRAIN)[0]

            # Execute given action in environment
            reward, next_state, done = self.env.step(action)

            # Pass observed transition to the worker
            self.worker.observe([state], [action], [reward], [next_state], [done])

            # Reset environment when episode ends
            if done:
                self.env.reset()

        # Update worker's progress
        self.workers_steps[self.worker.worker_id] += num_steps

        # Return result
        return TrainResult(self.batch_steps, timer() - start)


class EvalProcess(mp.Process):
    """
    Process which evaluates an agent.
    """

    def __init__(self, env_fn_serialized, agent, seed, train_steps, eval_every_sec, eval_episodes, goal_serialized,
                 workers_steps, stop_flag, result_queue):
        """
        Initialize evaluation process.

        :param env_fn_serialized: serialized function to create an environment
        :param agent: agent to evaluate
        :param seed: seed
        :param train_steps: number of training steps
        :param eval_every_sec: evaluate agent every `eval_every_sec` seconds
        :param eval_episodes: number of episode to evaluate agent for
        :param goal_serialized: serialized goal which can terminate training if it is reached
        :param workers_steps: array for storing workers' current step
        :param stop_flag: flag indicating that training should be terminated
        :param result_queue: queue where result should be write
        """
        super(EvalProcess, self).__init__()

        self.env = deserialize(env_fn_serialized)()
        self.agent = agent
        self.seed = seed
        self.train_steps = train_steps
        self.eval_every_sec = eval_every_sec
        self.eval_episodes = eval_episodes
        self.goal = deserialize(goal_serialized)
        self.workers_steps = workers_steps
        self.stop_flag = stop_flag
        self.result_queue = result_queue

    def run(self):
        """
        Run process.

        :return: None
        """
        # Set random seed for this process
        set_seed(self.seed)

        # Create eval agent
        eval_agent = copy.deepcopy(self.agent)

        eval_results = []

        while not self.stop_flag.is_set():
            # Wait for evaluation
            self.__wait_for_eval__()

            # Find out current step
            current_step = sum(self.workers_steps)

            # Copy current agent's state to eval agent
            eval_agent.model.load_state_dict(self.agent.model.state_dict())

            # Evaluate agent for given number of episodes
            result = self.__eval__(eval_agent, self.eval_episodes)

            # Store evaluation result
            eval_results.append(result)

            # Log evaluation result
            log_eval_result(current_step, result)

            # If termination condition passed given evaluation result, finish training
            if self.goal and self.goal(result):
                logger.info("")
                logger.info("Termination condition passed")
                logger.info("")
                self.stop_flag.set()

            # If workers reached total number of training steps, finish training
            if self.__workers_finished__():
                logger.info("")
                self.stop_flag.set()

        # Put result to the queue
        self.result_queue.put(RunResult([], eval_results))

    def __eval__(self, eval_agent, num_episodes):
        """
        Evaluate agent for given number of episodes.

        :param eval_agent: agent
        :param num_episodes: number of episodes
        :return: None
        """
        result = EvalResult()

        start = timer()

        for episode in range(num_episodes):
            episode_reward = 0

            # Reset an environment before episode
            state = self.env.reset()

            while not self.env.is_terminal():
                # Get agent's action
                action = eval_agent.act([state], RunPhase.EVAL)[0]

                # Execute given action in environment
                reward, next_state, done = self.env.step(action)

                episode_reward += reward

                # Update state
                state = next_state

            # Add episode result
            result.add_result(EvalEpisodeResult(
                reward=episode_reward,
                steps=self.env.state.step,
                has_won=self.env.has_won()))

        result.time = timer() - start

        return result

    def __workers_finished__(self):
        """
        Return True if workers finished training.

        :return: True if workers finished training
        """
        return sum(self.workers_steps) >= self.train_steps

    def __wait_for_eval__(self):
        """
        Wait for next evaluation.

        :return: None
        """
        seconds = 0

        while True:
            time.sleep(1)
            seconds += 1

            if self.__workers_finished__() or seconds >= self.eval_every_sec:
                break
