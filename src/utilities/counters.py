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

        self.measures = []

        logging.basicConfig()
        self.logger = logging.getLogger(self.name)
        if ('verbose', True) in kwargs.items():
            self.logger.setLevel(logging.DEBUG)

    def __repr__(self):
        return f'The cumulated time of {self.name} is {self.total_time}'

    def __call__(self, type_=None):
        if type_ == 'last':
            return self.measures[-1]
        elif type_ == 'mean':
            return self.average_time
        else:
            return self.total_time

    def start(self):
        assert ~self.counting, 'There is already a timer running'
        self.count = time()
        self.counting = True
        self.logger.info(f'Started count for {self.name}')

    def stop(self):
        assert self.counting, 'The timer is stopped'
        self.counting = False
        self.measures.append(time() - self.count)
        self.total_time += self.measures[-1]
        self.number_of_measures += 1
        self.average_time = self.total_time / self.number_of_measures
        self.logger.info(f'Stopped count for {self.name}, time vas {time() - self.count}\n')


class TallyCounter:
    def __init__(self, name):
        self.name = name
        self.count_ = 0
        self.list = []

    def __repr__(self):
        return f'The current count is {self.count_}'

    def __call__(self, *args, **kwargs):
        return self.count_

    def reset(self):
        self.count_ = 0
        self.list = []

    def count(self):
        self.count_ += 1

    def add(self, a):
        self.count()
        self.list.append(a)


class TallyMeasurer:
    def __init__(self, name):
        self.name = name
        self.count_ = 0
        self.number_of_measures = 0
        self.measures = []
        self.average = 0

    def __repr__(self):
        return f'The current count is {self.count_}'

    def __call__(self, type_=None):
        ell = self.measures
        mean = self.average
        n = self.number_of_measures
        if type_ == 'last':
            return ell[-1]
        elif type_ == 'mean':
            return mean
        elif type_ == 'std':
            return 1 / (n - 1) * sum(map(lambda x: (x - mean)**2, ell))

    def reset(self):
        self.count_ = 0
        self.number_of_measures = 0
        self.measures = []

    def measure(self, value):
        self.count_ += value
        self.measures.append(value)
        self.number_of_measures += 1
        self.average = self.count_ / self.number_of_measures

    def add(self, a):
        self.measure(a)
        self.measures.append(a)
