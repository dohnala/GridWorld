from agent.dqn_agent import DQNAgent
from agent.encoder import OneHotEncoder
from agent.model import NNModel
from agent.policy import EpsilonGreedyPolicy
from experiments.experiment import Experiment


class FindTreasureV0(Experiment):
    """
    Experiment of DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__("find_treasure_v0")

    def create_agent(self, env):
        encoder = OneHotEncoder(env.width, env.height)
        model = NNModel(encoder, env.num_actions)

        return DQNAgent(env, model,
                        learning_rate=0.01,
                        discount=0.9,
                        exploration_policy=EpsilonGreedyPolicy(0.5, 0.01, 150),
                        n_step=8)


if __name__ == "__main__":
    FindTreasureV0().run()
