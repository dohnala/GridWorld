from agents import NStepAgent
from models import QNstepModel
from policies import GreedyPolicy


class NStepDQNAgent(NStepAgent):
    """
    N-step Deep-Q-Network agent.
    """

    def __init__(self, env, encoder, network, optimizer, discount, exploration_policy, n_step=1):
        # Create model
        model = QNstepModel(encoder.shape(),
                            env.num_actions,
                            discount,
                            network)

        super(NStepDQNAgent, self).__init__("DQN agent",
                                            env,
                                            encoder,
                                            model,
                                            optimizer,
                                            exploration_policy,
                                            GreedyPolicy(),
                                            n_step)
