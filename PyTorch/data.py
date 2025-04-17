import collections
import numpy
import random


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))


class PER:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        z = zip(*self.buffer)
        delta = z[-1]
        p = delta / sum(delta)
        return numpy.random.choice(z, batch_size, p=p)
