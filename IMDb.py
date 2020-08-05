import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# num_words=10000 - кол-во слов, воторое будет записано из
# наиболее часто встерчающихся в обучающем наборе отзывов
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['acc'])

model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=512,)
results = model.evaluate(x_test, y_test)
print(results)

# [0.2981145977973938, 0.8807200193405151] with 3 Dense(16, 16, 1) & binary_crossentropy
# [0.08758426457643509, 0.8801199793815613] with 3 Dense(16, 16, 1) & mse
# [0.2835472524166107, 0.8873199820518494] with two Dense(16, 1) & binary_crossentropy
# [0.3737928569316864, 0.871399998664856] with 1 Dense(1) & binary_crossentropy
# [0.3278887867927551, 0.8740400075912476] with 3 Dense(64, 32, 1) & binary_crossentropy
# [0.09284312278032303, 0.8740800023078918] with 3 Dense(32, 16, 1) & mse
