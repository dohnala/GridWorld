from agents import Agent


class AsyncAgent(Agent):
    """
    Agent which is trained by asynchronous workers which share model.
    """

    def __init__(self, **kwargs):
        super(AsyncAgent, self).__init__(**kwargs)

        # Move model parameters to shared memory
        self.model.share_memory()

        # Set parameters of shared model to optimizer
        self.optimizer.set_shared_parameters(self.model.parameters())

        # Move optimizer parameters to shared memory
        self.optimizer.share_memory()

    def observe(self, state, action, reward, next_state, done):
        pass

    def create_workers(self, num_workers):
        """
        Create given number of workers.

        :param num_workers: number of workers
        :return: list of workers
        """
        return [self.__create_worker__(i) for i in range(num_workers)]

    def __create_worker__(self, worker_id):
        """
        Create a worker.

        :param worker_id: worker id
        :return: worker
        """
        pass


class WorkerAgent(Agent):
    """
    Worker agent which asynchronously updates a shared model.
    """

    def __init__(self, shared_model, **kwargs):
        """
        Initialize worker.

        :param shared_model: shared model
        :param kwargs: kwargs
        """
        super(WorkerAgent, self).__init__(**kwargs)

        self.shared_model = shared_model

    def __update_model__(self, transitions):
        super(WorkerAgent, self).__update_model__(transitions)

        # Copy shared model to worker model
        self.model.load_state_dict(self.shared_model.state_dict())
