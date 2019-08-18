from keras.datasets import imdb
import numpy as np
from keras import models, optimizers
from keras import layers
from keras import losses
from keras import metrics

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 数据集长度，每个评论维度10000
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # one-hot
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')  # 向量化标签数据
y_test = np.asarray(test_labels).astype('float32')

# model = models.Sequential()
#
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

# model validation
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results[1])

# f = open('history10000.txt', 'w')
# f.write(str(history.history))
# f.close()
