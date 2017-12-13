import copy
import os
import time
from timeit import default_timer as timer

import torch.multiprocessing as mp

from execution import Runner
from execution.result import RunResult, log_eval_result
from utils.logging import logger


class AsyncRunner(Runner):
    """
    Asynchronous runner implementation.
    """

    def __init__(self, env_fn, agent, num_workers, seed=None):
        """
        Initialize agent.

        :param env_fn: function to create environment
        :param agent: agent
        :param num_workers: number of workers
        :param seed: random seed
        """
        super(AsyncRunner, self).__init__(env_fn, agent, seed)

        self.num_workers = num_workers

    def train(self, max_steps, eval_every_sec, eval_episodes, goal=None):
        """
        Train agent for given number of steps.

        :param max_steps: maximum steps to train agent
        :param eval_every_sec: evaluate agent every `eval_every_sec` seconds
        :param eval_episodes: number of episode to evaluate agent for
        :param goal: goal which can terminate training if it is reached
        :return: result
        """

        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Flag indicating that training is finished
        stop_flag = mp.Event()

        # Max number of steps for each worker
        workers_max_steps = int(max_steps / self.num_workers)

        # Array with workers' current steps
        workers_steps = mp.Array('i', self.num_workers)

        # Queue where the final result is put
        result_queue = mp.Queue()

        processes = []

        start = timer()

        # Create evaluation process
        p = mp.Process(target=self.__eval__, args=(CloudpickleWrapper(self.env_fn), self.agent, max_steps,
                                                   eval_every_sec, eval_episodes, goal, stop_flag, workers_steps,
                                                   result_queue))
        p.start()
        processes.append(p)

        # Create worker processes
        for worker in self.agent.create_workers(self.num_workers):
            p = mp.Process(target=self.__train_worker__, args=(CloudpickleWrapper(self.env_fn), worker,
                                                               workers_max_steps, stop_flag, workers_steps))
            p.start()
            processes.append(p)

        # Wait until all processes finish execution
        for process in processes:
            process.join()

        # Get result from queue
        result = result_queue.get()
        result.train_time = timer() - start

        return result

    def __train_worker__(self, env_fn_wrapper, worker, max_steps, stop_flag, workers_steps, batch_steps=10):

        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + worker.worker_id)

        # Create environment
        env = env_fn_wrapper.o()

        # Initialize worker's current step
        workers_steps[worker.worker_id] = 0

        # Train until stop flag is set or number of training steps is reached
        while not stop_flag.is_set() and workers_steps[worker.worker_id] < max_steps:
            # Train worker for batch steps
            self.__train_steps__(env, worker, batch_steps)

            # Update worker's progress
            workers_steps[worker.worker_id] += batch_steps

    def __eval__(self, env_fn_wrapper, agent, max_steps, eval_every_sec, eval_episodes, goal, stop_flag,
                 workers_progress, result_queue):

        # Return True if all workers have finished training
        def workers_finished():
            return sum(workers_progress) >= max_steps

        # Sleep while checking if workers already finished training
        def wait_for_eval():
            seconds = 0

            while True:
                time.sleep(1)
                seconds += 1

                if workers_finished() or seconds >= eval_every_sec:
                    break

        # Set random seed for this process
        if self.seed:
            self.__set_seed__(self.seed + self.num_workers)

        # Create environment
        env = env_fn_wrapper.o()

        # Create eval agent
        eval_agent = copy.deepcopy(agent)

        eval_results = []

        while not stop_flag.is_set():
            # Wait for evaluation
            wait_for_eval()

            # Find out current step
            current_step = sum(workers_progress)

            # Copy current agent's state to eval agent
            eval_agent.model.load_state_dict(agent.model.state_dict())

            # Evaluate agent for given number of episodes
            result = self.__eval_episodes__(env, eval_agent, eval_episodes)

            # Store evaluation result
            eval_results.append(result)

            # Log evaluation result
            log_eval_result(current_step, result)

            # If termination condition passed given evaluation result, finish training
            if goal and goal(result):
                logger.info("")
                logger.info("Termination condition passed")
                logger.info("")
                stop_flag.set()

            # If workers reached total number of training steps, finish training
            if workers_finished():
                logger.info("")
                stop_flag.set()

        # Put result to the queue
        result_queue.put(RunResult([], eval_results))


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize object (otherwise multiprocessing tries to use pickle).
    """

    def __init__(self, o):
        self.o = o

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
