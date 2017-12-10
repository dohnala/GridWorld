from agents import Agent
from models import Model


class AsyncAgent(Agent):
    """
    Agent which is trained by asynchronous workers which share model.
    """

    def __init__(self, **kwargs):
        super(AsyncAgent, self).__init__(**kwargs)

        # Move model parameters to shared memory
        self.model.share_memory()

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

    def __init__(self, worker_id, shared_model, **kwargs):
        """
        Initialize worker.

        :param worker_id: worker id
        :param shared_model: shared model
        :param kwargs: kwargs
        """
        assert type(worker_id) is int, "worker_id is not valid"
        assert isinstance(shared_model, Model), "shared_model is not valud"

        self.worker_id = worker_id
        self.shared_model = shared_model

        super(WorkerAgent, self).__init__(**kwargs)

        # Copy shared model after worker is initialized
        self.__copy_shared_model__()

    def __create_optimizer__(self, optimizer_creator):
        return optimizer_creator.create(self.shared_model.parameters())

    def __update_model__(self, transitions):
        super(WorkerAgent, self).__update_model__(transitions)

        # Copy shared model after each update
        self.__copy_shared_model__()

    def __copy_shared_model__(self):
        """
        Copy shared model to local worker model.

        :return: None
        """
        self.model.load_state_dict(self.shared_model.state_dict())
