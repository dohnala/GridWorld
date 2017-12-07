from execution.result import AverageRunResult, log_run_result
from utils.logging import logger


class AverageRunner:
    """
    Runner which averages results across multiple runs.
    """

    def __init__(self, run_method):
        """
        Initialize runner.

        :param run_method: method to run multiple times
        """
        self.run_method = run_method

    def run(self, num_runs):
        """
        Run run method given times.

        :param num_runs: number of runs
        :return: average result across runs
        """
        results = []

        for i in range(num_runs):
            logger.info("# Run {}/{}".format(i + 1, num_runs))
            logger.info("")

            # Run method
            result = self.run_method()

            # Store result
            results.append(result)

            log_run_result(result)
            logger.info("-" * 150)

        # Create average run result
        result = AverageRunResult(results)

        return result
