from agents import MemoryAgentConfig, MemoryAgent
from models import QModelConfig, QModel
from policies import GreedyPolicy


class DQNAgentConfig(MemoryAgentConfig, QModelConfig):
    """
    DQN agent's configuration.
    """

    def __init__(self, num_actions, encoder, optimizer, policy, capacity, batch_size, network, discount,
                 target_sync=None, double_q=False, use_cuda=False):
        """
        Initialize configuration.

        :param num_actions: number of actions
        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param policy: policy used in training phase
        :param capacity: capacity of the memory
        :param batch_size: batch used to sample transitions from memory
        :param network: network used by model
        :param discount: discount factor used by model
        :param target_sync: after how many updates target network should be synced
        :param double_q: use double q learning
        :param use_cuda: use GPU
        """
        MemoryAgentConfig.__init__(self, encoder, optimizer, policy, GreedyPolicy(), capacity, batch_size)
        QModelConfig.__init__(self, encoder.shape(), num_actions, network, discount, target_sync, double_q, use_cuda)


class DQNAgent(MemoryAgent):
    """
    DQN agent.
    """

    def __init__(self, config):
        """
        Initialize agent.

        :param config: agent's configuration
        """
        super(DQNAgent, self).__init__(
            name="DQN agent",
            model=QModel(config=config),
            config=config)
