from time import time
import logging


class Timer:
    def __init__(self, name='Timer', **kwargs):
        self.name = name
        self.total_time = 0
        self.average_time = 0

        self.number_of_measures = 0
        self.count = 0
        self.counting = False

        logging.basicConfig()
        self.logger = logging.getLogger(self.name)
        if ('verbose', True) in kwargs.items():
            self.logger.setLevel(logging.DEBUG)

    def __repr__(self):
        return f'The cumulated time of {self.name} is {self.total_time}'

    def start(self):
        assert ~self.counting, 'There is already a timer running'
        self.count = time()
        self.counting = True
        self.logger.info(f'Started count for {self.name}')

    def stop(self):
        assert self.counting, 'The timer is stopped'
        self.counting = False
        self.total_time += time() - self.count
        self.number_of_measures += 1
        self.average_time = self.total_time / self.number_of_measures
        self.logger.info(f'Stopped count for {self.name}, time vas {time() - self.count}\n')


class TallyCounter:
    def __init__(self, name):
        self.name = name
        self.Count = 0
        self.List = []

    def __repr__(self):
        return f'The current count is {self.Count}'

    def __call__(self, *args, **kwargs):
        return self.Count

    def reset(self):
        self.Count = 0
        self.List = []

    def count(self):
        self.Count += 1

    def add(self, a):
        self.count()
        self.List.append(a)


class TallyMeasurer:
    def __init__(self, name):
        self.name = name
        self.Count = 0
        self.number_of_measures = 0
        self.List = []

    def __repr__(self):
        return f'The current count is {self.Count}'

    def __call__(self, *args, **kwargs):
        if ('mean', True) in kwargs.items():
            return self.Count / self.number_of_measures
        else:
            return self.Count

    def reset(self):
        self.Count = 0
        self.number_of_measures = 0
        self.List = []

    def measure(self, value):
        self.Count += value
        self.number_of_measures += 1

    def add(self, a):
        self.measure(a)
        self.List.append(a)
