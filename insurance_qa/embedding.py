import itertools
import numpy as np

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T
# array([[0],
#        [1],
#        [1],
#        [1],
#        [1],
#        [0],
#        [0],
#        [0]], dtype=int32)

lemma = lambda x: x.strip().lower().split(' ') # split sentence function 
sentences_lemmatized = [lemma(sentence) for sentence in sentences]
# [['sam', 'is', 'red'], ['hannah', 'not', 'red'], ['hannah', 'is', 'green'], ['bob', 'is', 'green'], ['bob', 'not', 'red'], ['sam', 'not', 'green'], ['sarah', 'is', 'red'], ['sarah', 'not', 'green']]

words = set(itertools.chain(*sentences_lemmatized))
#set(['sarah', 'sam', 'hannah', 'is', 'green', 'not', 'bob', 'red'])

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words)) 
#{'sarah': 0, 'sam': 1, 'hannah': 2, 'is': 3, 'green': 4, 'not': 5, 'bob': 6, 'red': 7}
idx2word = list(words)
#['sarah', 'sam', 'hannah', 'is', 'green', 'not', 'bob', 'red']

# convert the sentences a numpy array
to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
# [[1, 3, 7], [2, 5, 7], [2, 3, 4], [6, 3, 4], [6, 5, 7], [1, 5, 4], [0, 3, 7], [0, 5, 4]]
sentences_array = np.asarray(sentences_idx, dtype='int32')
# array([[1, 3, 7],
       # [2, 5, 7],
       # [2, 3, 4],
       # [6, 3, 4],
       # [6, 5, 7],
       # [1, 5, 4],
       # [0, 3, 7],
       # [0, 5, 4]], dtype=int32)

# parameters for the model
sentence_maxlen = 3
n_words = len(words) #8
n_embed_dims = 3

# put together a model to predict
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.models import Model

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
#<tf.Tensor 'input_1:0' shape=(?, 3) dtype=int32>
input_embedding = Embedding(n_words, n_embed_dims)(input_sentence)
# <tf.Tensor 'embedding_1/Gather:0' shape=(?, 3, 3) dtype=float32>
color_prediction = SimpleRNN(1)(input_embedding)
#<tf.Tensor 'simple_rnn_1/TensorArrayReadV3:0' shape=(?, 1) dtype=float32>


predict_green = Model(input=[input_sentence], output=[color_prediction])
#<keras.engine.training.Model object at 0x7f31ef3d7ed0>
predict_green.compile(optimizer='sgd', loss='binary_crossentropy')

# fit the model to predict what color each person is
predict_green.fit([sentences_array], [is_green], nb_epoch=5000, verbose=1)
# embeddings = predict_green.layers[1].get_weights()

# print out the embedding vector associated with each word
# for i in range(n_words):
# 	print('{}: {}'.format(idx2word[i], embeddings[i]))

loss = predict_green.evaluate([sentences_array], [is_green], verbose=1)
print('Accuracy: %f' % (loss))
