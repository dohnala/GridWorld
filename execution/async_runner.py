import copy
import os
import time
from timeit import default_timer as timer
from utils.logging import logger

import torch.multiprocessing as mp

from execution import Runner
from execution.result import RunResult, TrainResult, log_eval_result


class AsyncRunner(Runner):
    """
    Asynchronous runner implementation.
    """

    def __init__(self, env, agent, num_workers, seed=None):
        """
        Initialize agent.

        :param env: environment
        :param agent: agent
        :param num_workers: number of workers
        :param seed: random seed
        """
        super(AsyncRunner, self).__init__(env, agent, seed)

        self.num_workers = num_workers

    def train(self, train_episodes, eval_episodes, eval_after_sec, goal=None):

        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Flag indicating that training is finished
        stop_flag = mp.Event()

        # Array with all workers' progress
        workers_progress = mp.Array('i', self.num_workers)

        # Queue where the final result is put
        result_queue = mp.Queue()

        processes = []

        start = timer()

        # Create evaluation process
        p = mp.Process(target=self.__eval__, args=(self.env, self.agent, train_episodes, eval_episodes, eval_after_sec,
                                                   goal, stop_flag, workers_progress, result_queue))
        p.start()
        processes.append(p)

        # Create worker processes
        for worker in self.agent.create_workers(self.num_workers):
            p = mp.Process(target=self.__train_worker__, args=(self.env, worker, train_episodes, stop_flag,
                                                               workers_progress))
            p.start()
            processes.append(p)

        # Wait until all processes finish execution
        for process in processes:
            process.join()

        # Get result from queue
        result = result_queue.get()
        result.train_time = timer() - start

        return result

    def __train_worker__(self, env, worker, train_episodes, stop_flag, workers_progress, batch=10):

        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + worker.worker_id)

            # Initialize worker's progress
            workers_progress[worker.worker_id] = 0

        # Train until stop flag is set or number of training episodes is reached
        while not stop_flag.is_set() and workers_progress[worker.worker_id] < train_episodes:
            # Train worker for batch episode
            self.__train_episodes__(env, worker, batch)

            # Update worker's progress
            workers_progress[worker.worker_id] += batch

    def __eval__(self, env, agent, train_episodes, eval_episodes, eval_after_sec, goal, stop_flag,
                 workers_progress, result_queue):

        # Return True if all workers have finished training
        def workers_finished():
            return all(workers_progress[worker_id] >= train_episodes for worker_id in range(len(workers_progress)))

        # Sleep while checking if workers already finished training
        def wait_for_eval():
            seconds = 0

            while True:
                time.sleep(1)
                seconds += 1

                if workers_finished() or seconds >= eval_after_sec:
                    break

        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + self.num_workers)

        # Create eval agent
        eval_agent = copy.deepcopy(agent)

        train_result = TrainResult(100)
        eval_results = []

        while not stop_flag.is_set():
            # Wait for evaluation
            wait_for_eval()

            # Find out current episode
            current_episode = sum(workers_progress)

            # Copy current agent's state to eval agent
            eval_agent.model.load_state_dict(agent.model.state_dict())

            # Evaluate agent for given number of episodes
            result = self.__eval_episodes__(env, eval_agent, eval_episodes)

            # Store evaluation result
            eval_results.append(result)

            # Log evaluation result
            log_eval_result(current_episode, result)

            # If termination condition passed given evaluation result, finish training
            if goal and goal(result):
                logger.info("")
                logger.info("Termination condition passed")
                logger.info("")
                stop_flag.set()

            # If agents reached total number of training episodes, finish training
            if workers_finished():
                logger.info("")
                stop_flag.set()

        # Put result to the queue
        result_queue.put(RunResult([train_result], eval_results))
