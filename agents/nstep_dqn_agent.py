from agents import NStepAgent, NStepAgentConfig
from models import NstepQModel, QModelConfig
from policies import GreedyPolicy


class NStepDQNAgentConfig(NStepAgentConfig, QModelConfig):
    """
    N-step DQN agent's configuration.
    """

    def __init__(self, encoder, optimizer, policy, n_step, network, discount, target_sync=None):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param policy: policy used in training phase
        :param n_step: how many steps are stored before updating the model
        :param network: network used by model
        :param discount: discount factor used by model
        :param target_sync: after how many steps target network should be synced
        """
        NStepAgentConfig.__init__(self, encoder, optimizer, policy, GreedyPolicy(), n_step, keep_last=True)
        QModelConfig.__init__(self, network, discount)

        if target_sync:
            assert type(target_sync) is int and target_sync > 0, "target_sync has to be integer greater than zero"

        self.target_sync = target_sync


class NStepDQNAgent(NStepAgent):
    """
    N-step DQN agent.
    """

    def __init__(self, num_actions, config):
        """
        Initialize agent.

        :param num_actions: number of actions
        :param config: agent's configuration
        """
        super(NStepDQNAgent, self).__init__(
            name="N-step DQN agent",
            model=NstepQModel(
                input_shape=config.encoder.shape(),
                num_actions=num_actions,
                target_sync=config.target_sync,
                config=config),
            config=config)
