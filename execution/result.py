import numpy as np
from collections import deque


class TrainResult:
    def __init__(self, rolling_mean_window):
        self.num_episodes = 0
        self.rewards_per_episode = deque(maxlen=rolling_mean_window)
        self.steps_per_episode = deque(maxlen=rolling_mean_window)
        self.wins = deque(maxlen=rolling_mean_window)
        self.losses_per_episode = deque(maxlen=rolling_mean_window)
        self.time = 0

    def add_result(self, reward, steps, has_won, loss=None):
        self.num_episodes += 1
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(steps)
        self.wins.append(1 if has_won else 0)
        self.losses_per_episode.append(loss if loss else np.nan)

    def get_accuracy(self):
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        return np.mean(self.steps_per_episode)

    def get_mean_loss(self):
        return np.mean(self.losses_per_episode)


class EvalResult:
    def __init__(self):
        self.num_episodes = 0
        self.rewards_per_episode = deque()
        self.steps_per_episode = deque()
        self.wins = deque()
        self.time = 0

    def add_result(self, reward, steps, has_won):
        self.num_episodes += 1
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(steps)
        self.wins.append(1 if has_won else 0)

    def get_accuracy(self):
        return 100.0 * (sum(self.wins) / len(self.wins))

    def get_mean_reward(self):
        return np.mean(self.rewards_per_episode)

    def get_mean_steps(self):
        return np.mean(self.steps_per_episode)


class RunResult:
    def __init__(self, train_results, eval_results):
        self.train_episodes = sum(result.num_episodes for result in train_results)
        self.eval_episodes = eval_results[-1].num_episodes
        self.accuracy = eval_results[-1].get_accuracy()
        self.reward = eval_results[-1].get_mean_reward()
        self.steps = eval_results[-1].get_mean_steps()
        self.train_time = sum(result.time for result in train_results)
        self.eval_time = sum(result.time for result in eval_results)


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
