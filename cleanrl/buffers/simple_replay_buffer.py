import numpy as np
import random

class SimpleReplayBuffer( object):

    def __init__( self, observation_space, action_space, max_size = 10000, batch_size = 128):
        self._max_size = max_size
        self._batch_size = batch_size
        self._current_size = 0
        self._current_index = 0
        self._seed = 42

        self._Do = observation_space.shape
        self._Da = action_space.shape
        # TODO: Add support for disc. action space tasks too !

        self._observations = np.zeros([self._max_size] + list( self._Do))
        self._actions = np.zeros( [self._max_size]+ list( self._Da))
        self._rewards = np.zeros( self._max_size)
        self._terminals = np.zeros( self._max_size)
        self._next_observations = np.zeros( [self._max_size] + list( self._Do))

    def add_transition( self, observation, action, reward, terminal, next_observation):
        # TODO: Add verification on input shape

        self._observations[self._current_index] = np.reshape( observation, self._Do)
        self._actions[self._current_index] = action
        self._rewards[self._current_index] = reward
        self._terminals[self._current_index] = terminal
        self._next_observations[self._current_index] = np.reshape( next_observation, self._Do)

        self._current_index += 1
        self._current_index %= self._max_size
        self._current_size = max( self._current_index, self._current_size)

    def set_seed( self, seed = None):
        if seed is not None:
            # TODO: Numeric control on seed value
            self._seed = seed

        np.random.seed( self._seed)

    @property
    def size(self):
        return self._current_size

    @property
    def is_full(self):
        return self._current_index == self._max_size - 1

    @property
    def is_ready_for_sample( self, batch_size=None):
        if batch_size is None:
            return self.size >= self._batch_size

        return self.size >= batch_size

    def sample( self, batch_size = None):
        sample_batch_size = self._batch_size if batch_size is None else batch_size
        assert self.is_ready_for_sample, 'Not enough data to sample'

        batch_indices = np.random.randint(0, self.size, size=sample_batch_size)

        return self._observations[batch_indices], \
                self._actions[batch_indices], \
                self._rewards[batch_indices], \
                self._terminals[batch_indices], \
                self._next_observations[batch_indices]

    def store_episode(self, observations, actions, rewards, terminals, next_observations):
        for o, a, r, d, no in zip( observations, actions, rewards, terminals, next_observations):
            self.add_transition( o,a,r,d,no)
