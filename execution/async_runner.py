import multiprocessing as mp
import os
from timeit import default_timer as timer

from execution import Runner
from execution.result import RunResult, TrainResult


class AsyncRunner(Runner):
    """
    Asynchronous runner implementation.
    """
    def __init__(self, env_creator, agent_creator, num_workers):
        """
        Initialize agent.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
        :param num_workers: number of workers
        """
        super(AsyncRunner, self).__init__(env_creator, agent_creator)

        self.num_workers = num_workers

        self.running_flag = None
        self.episode_results_queue = None
        self.run_result_queue = None

    def __run__(self, run, train_episodes, eval_episodes, eval_after, termination_cond=None, after_run=None):
        # Set one thread per core
        os.environ['OMP_NUM_THREADS'] = '1'

        # Initialize flag and queues
        self.running_flag = mp.Event()
        self.episode_results_queue = mp.Queue()
        self.run_result_queue = mp.Queue()

        # Create agent for the run
        agent = self.agent_creator()

        processes = []

        start = timer()

        # Create evaluation process
        p = mp.Process(target=self.__eval__, args=(agent, train_episodes, eval_episodes, eval_after, termination_cond))
        p.start()
        processes.append(p)

        # Create worker processes
        for worker in agent.create_workers(self.num_workers):
            p = mp.Process(target=self.__train__, args=(worker, train_episodes))
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
        self.logger.info("-" * 150)

        # Call after run callback
        if after_run:
            after_run(run, agent)

        return result

    def __train__(self, worker, num_episodes):
        # Create new environment for the worker
        env = self.env_creator()

        episode = 0

        # Train until running_flag is set or number of training episodes is reached
        while not self.running_flag.is_set() and episode < num_episodes:
            # Train worker for one episode
            result = self.__train_episode__(env, worker)

            # Put result to the queue
            self.episode_results_queue.put((worker.worker_id, episode, result))

            episode += 1

    def __eval__(self, agent, train_episodes, eval_episodes, eval_after, termination_cond):
        # Create new environment and eval agent
        env = self.env_creator()
        eval_agent = self.agent_creator()

        # Hold worker's episode
        workers_episodes = {worker_id: 0 for worker_id in self.num_workers}

        current_episode = 0

        train_result = TrainResult(100)
        eval_results = []

        while True:
            # Get worker's result from queue
            worker_id, episode, result = self.episode_results_queue.get()

            current_episode += 1
            workers_episodes[worker_id] += 1

            train_result.add_result(result)

            if current_episode % eval_after == 0:
                # Copy agent's model to evaluation agent
                eval_agent.model.load_state_dict(agent.model.state_dict())

                # Evaluate agent for given number of episodes
                eval_result = self.__eval_episodes__(env, eval_agent, eval_episodes)

                # Log evaluation result
                self.__log_eval_result__(current_episode, eval_result)

                eval_results.append(eval_result)

                # If termination condition is defined and evaluated as True, break loop
                if termination_cond and termination_cond(eval_result):
                    self.logger.info("")
                    self.logger.info("Termination condition passed")
                    self.logger.info("")
                    break

            # If number of episodes exceed total number of training episodes, break loop
            if all(episodes >= train_episodes for episodes in workers_episodes.values()):
                break

        # Set running flag to stop worker processes
        self.running_flag.set()

        # Put run result to the queue
        self.run_result_queue.put(RunResult([train_result], eval_results))
