import random
from collections import deque
from agents import Agent, AgentConfig


class MemoryAgentConfig(AgentConfig):
    """
    Memory agent's configuration.
    """

    def __init__(self, encoder, optimizer, train_policy, eval_policy, capacity, batch_size, train_start=None):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param train_policy: policy used in training phase
        :param eval_policy: policy used in evaluation phase
        :param capacity: capacity of the memory
        :param batch_size: batch used to sample transitions from memory
        :param train_start: after how many steps should training using batch start
        """
        super(MemoryAgentConfig, self).__init__(encoder, optimizer, train_policy, eval_policy)

        assert type(capacity) is int and capacity > 0, "capacity has to be positive integer"
        assert type(batch_size) is int and batch_size > 0, "batch_size has to be positive integer"
        assert capacity >= batch_size, "capacity has to be greater or equals than batch_size"

        if train_start:
            assert type(train_start) is int and train_start > 0, "train_start has to be positive integer"
            assert train_start >= batch_size, "capacity has to be greater or equals than batch_size"

        self.capacity = capacity
        self.batch_size = batch_size

        # If train start is not specified, start training after memory is filled
        self.train_start = train_start if train_start else capacity


class MemoryAgent(Agent):
    """
    Memory agent which stores transitions in memory and used them for batch training.
    """
    def __init__(self, **kwargs):
        super(MemoryAgent, self).__init__(**kwargs)

        self.capacity = self.config.capacity
        self.batch_size = self.config.batch_size
        self.train_start = self.config.train_start
        self.memory = ReplayMemory(self.capacity)

    def __observe_transition__(self, transition):
        # Add transition into memory
        self.memory.add(transition)

        # If train should start
        if self.step >= self.train_start:
            # Sample batch from memory
            transitions = self.memory.sample(self.batch_size)

            # Update model using given transitions
            self.__update_model__(transitions)


class ReplayMemory:
    """
    Replay memory which stores a transitions.
    """
    def __init__(self, capacity):
        """
        Initialize replay memory with given capacity.

        :param capacity: capacity
        """
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition):
        """
        Add given transition into memory.

        :param transition: transition
        :return: None
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Sample transitions from memory using given batch size.

        :param batch_size: batch size
        :return: list of transitions
        """
        return random.sample(self.memory, batch_size)

    def size(self):
        """
        Return size of memory.

        :return: size of memory
        """
        return len(self.memory)