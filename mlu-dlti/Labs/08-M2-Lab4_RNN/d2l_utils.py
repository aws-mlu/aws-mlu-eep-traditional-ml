"""
Simple replacements for d2l utilities to avoid dependency issues.
"""
import time


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start_time = 0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)
