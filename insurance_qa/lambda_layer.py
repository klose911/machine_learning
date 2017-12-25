from keras.layers import Lambda, Input
from keras.models import Model

import numpy as np

input = Input(shape=(1,), dtype='float32')
double = Lambda(lambda x: 2 * x)(input)

model = Model(input=[input], output=[double])
model.compile(optimizer='sgd', loss='mse')

data = np.arange(5)
print(model.predict(data))
