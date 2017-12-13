from execution import Runner
from execution.result import RunResult, log_eval_result
from utils.logging import logger


class SyncRunner(Runner):
    """
    Synchronous runner implementation.
    """

    def __init__(self, env, agent, seed=None):
        """
        Initialize runner.

        :param env: environment
        :param agent: agent
        :param seed: random seed
        """
        super(SyncRunner, self).__init__(env, agent, seed)

    def train(self, max_steps, eval_every_steps, eval_episodes, goal=None):
        """
        Train agent for given number of steps.

        :param max_steps: maximum steps to train agent
        :param eval_every_steps: evaluate agent every `eval_every_steps` steps
        :param eval_episodes: number of episode to evaluate agent for
        :param goal: goal which can terminate training if it is reached
        :return: result
        """

        train_results = []
        eval_results = []

        current_step = 0

        while True:
            remaining_steps = max_steps - current_step
            num_steps = eval_every_steps if remaining_steps > eval_every_steps else remaining_steps

            # Reset environment
            self.env.reset()

            # Train agent
            train_result = self.__train_steps__(self.env, self.agent, num_steps)

            # Store training result
            train_results.append(train_result)

            # Update steps
            current_step += num_steps

            # Evaluate agent
            eval_result = self.__eval_episodes__(self.env, self.agent, eval_episodes)

            # Log evaluation result
            log_eval_result(current_step, eval_result)

            # Store evaluation result
            eval_results.append(eval_result)

            # If goal is defined and evaluated as True, break loop
            if goal and goal(eval_result):
                logger.info("")
                logger.info("Termination condition passed")
                logger.info("")
                break

            # If number of steps exceed total number of training steps, break loop
            if current_step >= max_steps:
                logger.info("")
                break

        # Return run result
        return RunResult(train_results, eval_results)
