import torch.multiprocessing as mp

from utils.multiprocessing import deserialize, serialize
from utils.seed import set_seed


class VecEnv:
    """
    Vectorized environment.
    """

    def get_states(self):
        """
        Return current states.

        :return: list of current states
        """
        pass

    def step(self, actions):
        """
        Perform one step with given actions return (reward, next_state, done) tuples.

        :param actions: list of actions to perform
        :return: list of (reward, next_state, done) tuples
        """
        pass

    def reset(self):
        """
        Reset all environments.

        :return: list of states
        """
        pass

    def close(self):
        """
        Close this vectorized environment.

        :return: None
        """
        pass


class SyncVecEnv(VecEnv):
    """
    Synchronous implementation of vectorized environment.
    """

    def __init__(self, env_fn, num_env, seed):
        """
        Initialize vectorized environment.

        :param env_fn: function to create environment
        :param num_env: number of environments.
        """
        self.envs = [env_fn() for _ in range(num_env)]

    def get_states(self):
        return [env.state for env in self.envs]

    def step(self, actions):
        # Perform actions
        results = [env.step(action) for (action, env) in zip(actions, self.envs)]

        # Convert results
        rewards, next_states, dones = zip(*results)

        # Reset environments if done flag is set
        for (i, done) in enumerate(dones):
            if done:
                self.envs[i].reset()

        return list(rewards), list(next_states), list(dones)

    def reset(self):
        return [env.reset() for env in self.envs]


class AsyncVecEnv(VecEnv):
    """
    Asynchronous implementation of vectorized environment.
    """

    def __init__(self, env_fn, num_env, seed):
        """
        Initialize vectorized environment.

        :param env_fn: function to create environment
        :param num_env: number of environments.
        """
        self.closed = False

        # Create pipes
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_env)])

        # Create processes
        self.processes = []

        for i in range(num_env):
            process_seed = seed + i if seed is not None else None
            process = mp.Process(target=self.worker,
                                 args=(serialize(env_fn), process_seed, self.work_remotes[i], self.remotes[i]))

            process.daemon = True
            process.start()

            self.processes.append(process)

        # Close pipes
        for remote in self.work_remotes:
            remote.close()

    def get_states(self):
        # Send state command
        for remote in self.remotes:
            remote.send(('state', None))

        # Return response
        return [remote.recv() for remote in self.remotes]

    def step(self, actions):
        # Send step command
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # Wait for response
        results = [remote.recv() for remote in self.remotes]

        # Convert results
        rewards, next_states, dones = zip(*results)

        return list(rewards), list(next_states), list(dones)

    def reset(self):
        # Send reset command
        for remote in self.remotes:
            remote.send(('reset', None))

        # Return response
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return

        # Send close command
        for remote in self.remotes:
            remote.send(('close', None))

        # Wait for processes
        for process in self.processes:
            process.join()

        self.closed = True

    def worker(self, env_fn_serialized, seed, remote, parent_remote):
        # Set random seed for this process
        set_seed(seed)

        # Close pipe
        parent_remote.close()

        # Create environment
        env = deserialize(env_fn_serialized)()

        while True:
            # Wait for data
            cmd, data = remote.recv()

            if cmd == 'state':
                # Return current state
                remote.send(env.state)
            elif cmd == 'step':
                # Perform action
                reward, next_state, done = env.step(data)

                # Reset environments if done flag is set
                if done:
                    env.reset()

                # Return observation
                remote.send((reward, next_state, done))
            elif cmd == 'reset':
                # Reset environment
                state = env.reset()

                remote.send(state)
            elif cmd == 'close':
                # Close pipe
                remote.close()

                break
            else:
                raise NotImplementedError
