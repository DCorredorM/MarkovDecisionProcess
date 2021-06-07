import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Flatten, InputLayer


class RLConfig:
	# Network default config
	net_init = 'uniform'
	net_config = [
		('conv2d', {'filters': 32, 'kernel_size': 8, 'strides': 4, 'activation': 'relu'}),
		('conv2d', {'filters': 64, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'}),
		('conv2d', {'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'}),
		('flatten', {}),
		('dense', {'units': 512, 'activation': 'relu'}),
		('dense', {'activation': 'linear'})
	]
	last_layer_activation = 'linear'
	input_data_type = tf.uint8

	# env config
	render_train = False
	render_test = False
	overwrite_render = True
	record = False
	high = 255.

	# output config
	output_path = "../results/q3_nature/"
	model_output = output_path + "model.weights"
	log_path = output_path + "log.txt"
	plot_output = output_path + "scores.png"

	# model and training config
	num_episodes_test = 50
	grad_clip = True
	clip_val = 10
	saving_freq = 250000
	log_freq = 50
	eval_freq = 250000
	record_freq = 250000
	soft_epsilon = 0.05

	# nature paper hyper params
	nsteps_train = 5000000
	batch_size = 32
	buffer_size = 500000  # 1000000
	target_update_freq = 10000
	gamma = 0.99
	learning_freq = 4
	state_history = 4
	skip_frame = 4
	lr_begin = 0.00025
	lr_end = 0.00005
	lr_nsteps = nsteps_train / 2
	eps_begin = 1
	eps_end = 0.1
	eps_nsteps = 1000000
	learning_start = 50000

	def __init__(self, **kwargs):
		for name, value in kwargs.items():
			self._set_attribute(name, value)

		# maps the string to layer objects
		for i, (lay, lay_kw) in enumerate(self.net_config):
			self.net_config[i] = (RLConfig.layer(lay), lay_kw)

	# noinspection PyMethodMayBeStatic
	def _set_attribute(self, name, value):
		if isinstance(value, str):
			exec(f'self.{name}="{value}"')
		else:
			exec(f'self.{name}={value}')

	@staticmethod
	def layer(text_layer: str):
		map_ = {'dense': Dense, 'conv2d': Conv2D, 'conv3b': Conv3D, 'flatten': Flatten, 'input': InputLayer}
		return map_[text_layer.lower()]
