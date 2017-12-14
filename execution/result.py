import numpy as np
from collections import deque
from utils.logging import logger


class TrainResult:
    def __init__(self, steps, time):
        self.steps = steps
        self.time = time

    
def log_train_result(result, current_episode, train_episodes):
    logger.info("Training {:4d}/{} - loss:{:>11f}, accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
        current_episode ,
        train_episodes,
        result.get_mean_loss(),
        result.get_accuracy(),
        result.get_mean_reward(),
        result.get_mean_steps()))


class EvalEpisodeResult:
    def __init__(self, reward, steps, has_won):
        self.reward = reward
        self.steps = steps
        self.has_won = has_won


class EvalResult:
    def __init__(self):
        self.num_episodes = 0
        self.rewards_per_episode = deque()
        self.steps_per_episode = deque()
        self.wins = deque()
        self.time = 0

    def add_result(self, eval_episode_result):
        self.num_episodes += 1
        self.rewards_per_episode.append(eval_episode_result.reward)
        self.steps_per_episode.append(eval_episode_result.steps)
        self.wins.append(1 if eval_episode_result.has_won else 0)

    def get_accuracy(self):
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        return np.mean(self.steps_per_episode)


def log_eval_result(current_step, result):
    logger.info("Evaluation at {:6d} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}".format(
        current_step,
        result.get_accuracy(),
        result.get_mean_reward(),
        result.get_mean_steps()))


class RunResult:
    def __init__(self, train_results, eval_results):
        self.train_steps = sum(result.steps for result in train_results)
        self.eval_episodes = eval_results[-1].num_episodes
        self.accuracy = eval_results[-1].get_accuracy()
        self.reward = eval_results[-1].get_mean_reward()
        self.steps = eval_results[-1].get_mean_steps()
        self.train_time = sum(result.time for result in train_results)
        self.eval_time = sum(result.time for result in eval_results)
        

def log_run_result(result):
    logger.info("Result - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}, train_time:{:5.2f}s".format(
        result.accuracy,
        result.reward,
        result.steps,
        result.train_time))


class AverageRunResult:
    def __init__(self, run_results):
        self.run_results = run_results
        self.num_runs = len(run_results)
        self.eval_episodes = run_results[0].eval_episodes
        self.accuracy_per_run = [result.accuracy for result in run_results]
        self.reward_per_run = [result.reward for result in run_results]
        self.steps_per_run = [result.steps for result in run_results]
        self.train_time_per_run = [result.train_time for result in run_results]

    def get_accuracy(self):
        return np.mean(self.accuracy_per_run)

    def get_mean_reward(self):
        return np.mean(self.reward_per_run)

    def get_mean_steps(self):
        return np.mean(self.steps_per_run)
    

def log_average_run_result(result):
    logger.info("# Run results")
    logger.info("")

    for i in range(result.num_runs):
        run_result = result.run_results[i]

        logger.info("Run {:2d} - accuracy:{:7.2f}%, reward:{:6.2f}, steps:{:6.2f}, train_time:{:5.2f}s".format(
            i + 1,
            run_result.accuracy,
            run_result.reward,
            run_result.steps,
            run_result.train_time))

    logger.info("")
    logger.info("# Average statistics")
    logger.info("")

    logger.info("Runs       - {}".format(result.num_runs))

    logger.info("Accuracy   - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
        np.mean(result.accuracy_per_run),
        np.min(result.accuracy_per_run),
        np.max(result.accuracy_per_run),
        np.var(result.accuracy_per_run)))

    logger.info("Reward     - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
        np.mean(result.reward_per_run),
        np.min(result.reward_per_run),
        np.max(result.reward_per_run),
        np.var(result.reward_per_run)))

    logger.info("Steps      - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
        np.mean(result.steps_per_run),
        np.min(result.steps_per_run),
        np.max(result.steps_per_run),
        np.var(result.steps_per_run)))

    logger.info("Train time - mean:{:7.2f}, min:{:7.2f}, max:{:7.2f}, var:{:7.2f}".format(
        np.mean(result.train_time_per_run),
        np.min(result.train_time_per_run),
        np.max(result.train_time_per_run),
        np.var(result.train_time_per_run)))

    logger.info("")
