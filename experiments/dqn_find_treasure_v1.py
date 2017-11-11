import torch.optim as optim

from agent.dqn_agent import DQNAgent
from agent.encoder import OneHotEncoder
from agent.model import NNModel
from agent.policy import EpsilonGreedyPolicy
from experiments.experiment import Experiment


class FindTreasureV1(Experiment):
    """
    Experiment of DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__("find_treasure_v1")

    def create_agent(self, env):
        encoder = OneHotEncoder(env.width, env.height, treasure_position=True)
        model = NNModel(encoder, env.num_actions, hidden_units=[128])

        return DQNAgent(env=env,
                        model=model,
                        optimizer=optim.Adam(model.parameters(), lr=0.002),
                        discount=0.95,
                        exploration_policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                        n_step=8,
                        sync_target=100)


if __name__ == "__main__":
    FindTreasureV1().run()
