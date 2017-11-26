from execution import Runner
from execution.result import RunResult


class SyncRunner(Runner):
    """
    Synchronous runner implementation.
    """

    def __init__(self, env_creator, agent_creator):
        """
        Initialize runner.

        :param env_creator: function to create environment
        :param agent_creator: function to create agent
        """
        super(SyncRunner, self).__init__(env_creator, agent_creator)

    def __run__(self, run, train_episodes, eval_episodes, eval_after, termination_cond=None, after_run=None):
        # Create env and agent for the run
        env = self.env_creator()
        agent = self.agent_creator()

        train_results = []
        eval_results = []

        current_episode = 0

        while True:
            remaining_episodes = train_episodes - current_episode
            num_episodes = eval_after if remaining_episodes > eval_after else remaining_episodes

            # Run training phase
            train_result = self.__train_episodes__(env, agent, num_episodes)

            # Store training result
            train_results.append(train_result)

            # Update episodes
            current_episode += num_episodes

            # Run evaluation phase
            eval_result = self.__eval_episodes__(env, agent, eval_episodes)

            # Log evaluation result
            self.__log_eval_result__(current_episode, eval_result)

            # Store evaluation result
            eval_results.append(eval_result)

            # If termination condition is defined and evaluated as True, break loop
            if termination_cond and termination_cond(eval_result):
                self.logger.info("")
                self.logger.info("Termination condition passed")
                self.logger.info("")
                break

            # If number of episodes exceed total number of training episodes, break loop
            if current_episode >= train_episodes:
                break

        # Create run result
        result = RunResult(train_results, eval_results)

        # Log run result
        self.__log_run_result__(result)
        self.logger.info("-" * 150)

        # Call after run callback
        if after_run:
            after_run(run, agent)

        return result
