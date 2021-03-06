from env.action import MoveUp, MoveDown, MoveRight, MoveLeft
from env.state import GridWorld, Agent, Treasure
from env.task import GridWorldTask


class FindTreasureTask(GridWorldTask):
    """
    Description:

    A WxH grid world with Agent and Treasure. Reward is earned by moving the Agent to the Treasure position.

    Initial state:
        - Agent at random position
        - Treasure at fixed or random position

    Possible actions:
        - 0 - MoveUp
        - 1 - MoveDown
        - 2 - MoveRight
        - 3 - MoveLeft

    Rewards:
        - Agent reaches the goal: 1
        - Agent doesn't reach the goal: -0.01

    End condition
        - Agent reaches the goal
        - Defined steps elapsed
    """

    def __init__(self, width, height, episode_length, treasure_position=None):
        super().__init__(width, height, self.generate_grid_world)

        self.episode_length = episode_length
        self.treasure_position = treasure_position

    def generate_grid_world(self):
        grid_world = GridWorld(self.width, self.height)

        if self.treasure_position:
            grid_world.add_object(Treasure(*self.treasure_position))
        else:
            grid_world.add_object(Treasure(*grid_world.get_random_free_position()))

        grid_world.add_agent(Agent(*grid_world.get_random_free_position()))

        return grid_world

    def get_actions(self):
        return [MoveUp(), MoveDown(), MoveRight(), MoveLeft()]

    def get_reward(self, state, action, next_state):
        if self.is_winning(next_state):
            return 1
        else:
            return -0.01

    def is_winning(self, state):
        return state.agent.is_at_any_object(state.get_objects_by_type(Treasure))

    def is_losing(self, state):
        return state.step == self.episode_length


class FindTreasureTaskV0(FindTreasureTask):
    """
    Description:

    A 9x9 grid world with Agent and Treasure. Reward is earned by moving the Agent to the position of Treasure.

    Initial state:
        - Agent at random position
        - Treasure at position (6, 7)

    Possible actions:
        - 0 - MoveUp
        - 1 - MoveDown
        - 2 - MoveRight
        - 3 - MoveLeft

    Rewards:
        - Agent reaches the goal: 1
        - Agent doesn't reach the goal: -0.01

    End condition
        - Agent reaches the goal
        - 60 steps elapsed
    """
    name = "find_treasure_v0"

    def __init__(self):
        super().__init__(width=9, height=9, episode_length=60, treasure_position=(6, 7))

    def __str__(self):
        return self.name


class FindTreasureTaskV1(FindTreasureTask):
    """
    Description:

    A 9x9 grid world with Agent and Treasure. Reward is earned by moving the Agent to the position of Treasure.

    Initial state:
        - Agent at random position
        - Treasure at random position

    Possible actions:
        - 0 - MoveUp
        - 1 - MoveDown
        - 2 - MoveRight
        - 3 - MoveLeft

    Rewards:
        - Agent reaches the goal: 1
        - Agent doesn't reach the goal: -0.01

    End condition
        - Agent reaches the goal
        - 60 steps elapsed
    """
    name = "find_treasure_v1"

    def __init__(self):
        super().__init__(width=9, height=9, episode_length=60)

    def __str__(self):
        return self.name
