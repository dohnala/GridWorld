from agents import NStepAgent, NStepAgentConfig
from models import NstepQModel, NstepQModelConfig
from policies import GreedyPolicy


class NStepDQNAgentConfig(NStepAgentConfig, NstepQModelConfig):
    """
    N-step DQN agent's configuration.
    """

    def __init__(self, encoder, optimizer, policy, n_step, network, discount):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param policy: policy used in training phase
        :param n_step: how many steps are stored before updating the model
        :param discount: discount factor used by model
        :param network: network used by model
        """
        NStepAgentConfig.__init__(self, encoder, optimizer, policy, GreedyPolicy(), n_step)
        NstepQModelConfig.__init__(self, network, discount)


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
            name="DQN agent",
            model=NstepQModel(
                input_shape=config.encoder.shape(),
                num_actions=num_actions,
                config=config),
            config=config)
