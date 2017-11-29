from agents import MemoryAgentConfig, MemoryAgent
from models import QModelConfig, QModel
from policies import GreedyPolicy


class DQNAgentConfig(MemoryAgentConfig, QModelConfig):
    """
    DQN agent's configuration.
    """

    def __init__(self, encoder, optimizer, policy, capacity, batch_size, network, discount, target_sync=None):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param policy: policy used in training phase
        :param capacity: capacity of the memory
        :param batch_size: batch used to sample transitions from memory
        :param network: network used by model
        :param discount: discount factor used by model
        :param target_sync: after how many steps target network should be synced
        """
        MemoryAgentConfig.__init__(self, encoder, optimizer, policy, GreedyPolicy(), capacity, batch_size)
        QModelConfig.__init__(self, network, discount)

        self.target_sync = target_sync


class DQNAgent(MemoryAgent):
    """
    DQN agent.
    """

    def __init__(self, num_actions, config):
        """
        Initialize agent.

        :param num_actions: number of actions
        :param config: agent's configuration
        """
        super(DQNAgent, self).__init__(
            name="DQN agent",
            model=QModel(
                input_shape=config.encoder.shape(),
                num_actions=num_actions,
                target_sync=config.target_sync,
                config=config),
            config=config)
