import collections
import numpy


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        delta_batch = numpy.array([self.buffer[i][-1].detach() for i in range(len(self.buffer))])
        delta_batch -= delta_batch.max()
        return zip(*[self.buffer[i] for i in
                     numpy.random.choice(len(self.buffer), batch_size, p=delta_batch / delta_batch.sum())])
