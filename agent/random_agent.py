import numpy as np

from agent.agent import Agent


class RandomAgent(Agent):
    """
    Agent which selects random actions.
    """

    def __init__(self, env):
        super().__init__(env)

    def __select_action__(self, state):
        # Select random action
        return np.random.choice(self.env.get_actions())
