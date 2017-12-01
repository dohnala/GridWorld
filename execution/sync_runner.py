from execution import Runner
from execution.result import RunResult, log_eval_result


class SyncRunner(Runner):
    """
    Synchronous runner implementation.
    """

    def __init__(self, env, agent, seed=1):
        """
        Initialize runner.

        :param env: environment
        :param agent: agent
        :param seed: random seed
        """
        super(SyncRunner, self).__init__(env, agent, seed)

    def train(self, train_episodes, eval_episodes, eval_after, goal=None):

        train_results = []
        eval_results = []

        current_episode = 0

        while True:
            remaining_episodes = train_episodes - current_episode
            num_episodes = eval_after if remaining_episodes > eval_after else remaining_episodes

            # Run training phase
            train_result = self.__train_episodes__(self.env, self.agent, num_episodes)

            # Store training result
            train_results.append(train_result)

            # Update episodes
            current_episode += num_episodes

            # Run evaluation phase
            eval_result = self.__eval_episodes__(self.env, self.agent, eval_episodes)

            # Log evaluation result
            log_eval_result(self.logger, current_episode, eval_result)

            # Store evaluation result
            eval_results.append(eval_result)

            # If goal is defined and evaluated as True, break loop
            if goal and goal(eval_result):
                self.logger.info("")
                self.logger.info("Termination condition passed")
                self.logger.info("")
                break

            # If number of episodes exceed total number of training episodes, break loop
            if current_episode >= train_episodes:
                self.logger.info("")
                break

        # Return run result
        return RunResult(train_results, eval_results)
