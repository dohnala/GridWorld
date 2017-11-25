class GridWorldEnv:
    """
    Grid world environment which agents can interact with.
    """

    def __init__(self, task):
        """
        Initialize grid world environment for given task.

        :param task: task
        """
        self.task = task
        self.width = self.task.width
        self.height = self.task.height
        self.actions = list(range(len(self.task.get_actions())))
        self.num_actions = len(self.actions)
        self.state = None
        self.observers = []

        self.reset()

    def step(self, action_index):
        """
        Perform one step with given action index and return (reward, next_state, done) tuple.

        :param action_index: index of action for get_actions() list
        :return: (reward, next_state, done) tuple
        """
        state = self.state

        if self.task.is_terminal(state):
            raise Exception("Cannot perform action in terminal state.")

        if not (0 <= action_index < self.num_actions):
            raise Exception("Cannot perform action with index " + action_index + ".")

        action = self.task.get_actions()[action_index]

        next_state = action.apply(state)
        reward = self.task.get_reward(state, action, next_state)

        self.state = next_state

        # notify observers
        for observer in self.observers:
            observer.on_step(action, reward, next_state)

            if self.is_terminal():
                observer.on_end(self.has_won())

        return reward, next_state, self.is_terminal()

    def is_terminal(self):
        """
        Return if current state is terminal.

        :return: if current state is terminal
        """
        return self.task.is_terminal(self.state)

    def has_won(self):
        """
        Return if agent has won.

        :return: if agent has won
        """
        return self.task.is_winning(self.state)

    def has_lost(self):
        """
        Return if agent has lost.

        :return: if agent has lost
        """
        return self.task.is_losing(self.state)

    def reset(self):
        """
        Reset environment and return current state.

        :return current state
        """
        self.state = self.task.get_start_state()

        # notify observers
        for observer in self.observers:
            observer.on_reset(self.state)

        return self.state

    def add_observer(self, observer):
        """
        Add observer which can listen to changes of this environment.

        :param observer: observer
        """
        self.observers.append(observer)


class EnvObserver:
    """
    Observer which can listen for grid world environment changes.
    """

    def on_step(self, action, reward, next_state):
        """
        Handle step in environment.

        :param action: action taken
        :param reward: reward gained
        :param next_state: next state
        """
        pass

    def on_end(self, is_winning):
        """
        Handle end of episode.

        :param is_winning: is agent winning
        """
        pass

    def on_reset(self, state):
        """
        Handle environment reset.

        :param state: new state
        """
        pass
