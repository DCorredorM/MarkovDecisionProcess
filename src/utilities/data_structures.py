from queue import Queue
import random


def sample_n_unique(sampling_f, n):
	"""Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
	res = []
	while len(res) < n:
		candidate = sampling_f()
		if candidate not in res:
			res.append(candidate)
	return res


class RollingWindow(Queue):
	def __init__(self, maxsize):
		super().__init__(maxsize=maxsize)
		self.num_in_buffer = 0

	def __repr__(self):
		return f'{self.as_list()}'

	def add(self, element):
		if self.full():
			self.get_nowait()
		self.put_nowait(element)
		self.num_in_buffer += 1

	def as_list(self):
		with self.mutex:
			return list(self.queue)

	def can_sample(self, batch_size):
		"""Returns true if `batch_size` different transitions can be sampled from the buffer."""
		return batch_size + 1 <= self.num_in_buffer
