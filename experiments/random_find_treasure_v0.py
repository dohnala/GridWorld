from agent.random_agent import RandomAgent
from experiments.experiment import Experiment


class FindTreasureV0(Experiment):
    """
    Experiment of Random agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__("find_treasure_v0")

    def create_agent(self, env):
        return RandomAgent(env)


if __name__ == "__main__":
    FindTreasureV0().run()
