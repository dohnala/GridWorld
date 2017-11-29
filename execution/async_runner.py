import torch.multiprocessing as mp
import os
from timeit import default_timer as timer

import time

from execution import Runner
from execution.result import RunResult, TrainResult


class AsyncRunner(Runner):
    """
    Asynchronous runner implementation.
    """
    def __init__(self, env_creator, agent_creator, num_workers, seed=1):
        """
        Initialize agent.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
        :param num_workers: number of workers
        :param seed: random seed
        """
        super(AsyncRunner, self).__init__(env_creator, agent_creator, seed)

        self.num_workers = num_workers

        self.stop_flag = None
        self.train_barrier = None
        self.run_result_queue = None
        self.agent_progress = None

    def __train__(self, run, train_episodes, eval_episodes, eval_after, termination_cond=None, after_run=None):
        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Initialize flag and queues
        self.stop_flag = mp.Event()
        self.eval_barrier = mp.Barrier(self.num_workers + 1)
        self.run_result_queue = mp.Queue()
        self.agent_progress = mp.Array('i', self.num_workers)

        # Create agent for the run
        agent = self.agent_creator()

        processes = []

        start = timer()

        # Create evaluation process
        p = mp.Process(target=self.__eval__, args=(agent, train_episodes, eval_episodes, termination_cond))
        p.start()
        processes.append(p)

        # Create worker processes
        for worker in agent.create_workers(self.num_workers):
            p = mp.Process(target=self.__train_worker__, args=(worker, train_episodes))
            p.start()
            processes.append(p)

        # Wait until all processes finish execution
        for process in processes:
            process.join()

        # Get result from queue
        result = self.run_result_queue.get()
        result.train_time = timer() - start

        # Log run result
        self.__log_run_result__(result)

        # Call after run callback
        if after_run:
            after_run(run, agent)

        self.logger.info("-" * 150)

        return result

    def __train_worker__(self, worker, train_episodes):
        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + worker.worker_id)

        # Create new environment for the worker
        env = self.env_creator()

        # Initialize agent's progress
        self.agent_progress[worker.worker_id] = 0

        # Train until stop flag is set or number of training episodes is reached
        while not self.stop_flag.is_set() and self.agent_progress[worker.worker_id] < train_episodes:
            # Train worker for one episode
            result = self.__train_episode__(env, worker)

            # Update agent's progress
            self.agent_progress[worker.worker_id] += 1

    def __eval__(self, agent, train_episodes, eval_episodes, termination_cond):
        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + self.num_workers)

        # Create new environment and eval agent
        env = self.env_creator()
        eval_agent = self.agent_creator()

        train_result = TrainResult(100)
        eval_results = []

        while not self.stop_flag.is_set():
            time.sleep(5)

            # Find out current episode
            current_episode = sum(self.agent_progress)

            # Copy current agent's state to eval agent
            eval_agent.model.load_state_dict(agent.model.state_dict())

            # Evaluate agent for given number of episodes
            result = self.__eval_episodes__(env, eval_agent, eval_episodes)

            # Store evaluation result
            eval_results.append(result)

            # Log evaluation result
            self.__log_eval_result__(current_episode, result)

            # If termination condition passed given evaluation result, finish training
            if termination_cond and termination_cond(result):
                self.logger.info("")
                self.logger.info("Termination condition passed")
                self.stop_flag.set()

            # If agents reached total number of training episodes, finish training
            if all(self.agent_progress[agent_id] >= train_episodes for agent_id in range(len(self.agent_progress))):
                self.stop_flag.set()

        # Put run result to the queue
        self.run_result_queue.put(RunResult([train_result], eval_results))
