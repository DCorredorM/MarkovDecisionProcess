from time import time


class Timer:
	def __init__(self, name='Timer'):
		self.name = name
		self.T = 0
		self.count = 0
		self.counting = False

	def __repr__(self):
		return f'The cumulated time of {self.name} is {self.T}'

	def start(self):
		assert ~self.counting, 'There is already a timer running'
		self.count = time()
		self.counting = True

	def stop(self):
		assert self.counting, 'The timer is stopped'
		self.counting = False
		self.T += time() - self.count


class Tally:
	def __init__(self, name):
		self.name = name
		self.Count = 0
		self.List = []

	def __repr__(self):
		return f'The current count is {self.Count}'

	def count(self):
		self.Count += 1

	def add(self, a):
		self.List.append(a)
