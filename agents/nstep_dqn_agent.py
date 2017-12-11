from agents import NStepAgent, NStepAgentConfig, AsyncAgent, WorkerAgent
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
        QModelConfig.__init__(self, network, discount, target_sync)


class NStepDQNAgent(NStepAgent, AsyncAgent):
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
                config=config),
            config=config)

        self.num_actions = num_actions

    def __create_worker__(self, worker_id, shared_model):
        return NStepDQNWorkerAgent(
            worker_id=worker_id,
            num_actions=self.num_actions,
            shared_model=shared_model,
            config=self.config)


class NStepDQNWorkerAgent(NStepAgent, WorkerAgent):
    """
    N-step DQN worker.
    """

    def __init__(self, worker_id, num_actions, shared_model, config):
        """
        Initialize worker.

        :param worker_id: worker id
        :param num_actions: number of actions
        :param shared_model: shared model
        :param config: worker's configuration
        """
        super(NStepDQNWorkerAgent, self).__init__(
            worker_id=worker_id,
            name="N-step DQN worker {}".format(worker_id),
            model=NstepQModel(
                input_shape=config.encoder.shape(),
                num_actions=num_actions,
                config=config),
            shared_model=shared_model,
            config=config)
