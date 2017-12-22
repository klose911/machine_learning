from keras.engine import Layer
from keras import initializations

# our layer will take input shape (nb_samples, 1)
class MultiplicationLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializations.get('glorot_uniform')
		super(MultiplicationLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# each sample should be a scalar
		assert len(input_shape) == 2 and input_shape[1] == 1
		self.multiplicand = self.init(input_shape[1:], name='multiplicand')

		# let Keras know that we want to train the multiplicand
		self.trainable_weights = [self.multiplicand]

	def get_output_shape_for(self, input_shape):
		# we're doing a scalar multiply, so we don't change the input shape
		assert input_shape and len(input_shape) == 2 and input_shape[1] == 1
		return input_shape

	def call(self, x, mask=None):
		# this is called during MultiplicationLayer()(input)
		return x * self.multiplicand

# test the model
from keras.layers import Input
from keras.models import Model

# input is a single scalar
input = Input(shape=(1,), dtype='int32')
multiply = MultiplicationLayer()(input)

model = Model(input=[input], output=[multiply])
model.compile(optimizer='sgd', loss='mse')

import numpy as np
input_data = np.arange(10)
output_data = 3 * input_data

model.fit([input_data], [output_data], nb_epoch=10)
print(model.layers[1].multiplicand.get_value())
# should be close to 3
