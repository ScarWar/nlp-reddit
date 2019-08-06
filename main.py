import os

import keras

from models.text_generator import PosTagDataGenerator


class Params:
    data_file = 'test.csv'
    data_base = 'data'
    data_file_path = os.path.join(data_base, data_file)
    batch_size = 32


config = Params()
# For resourceful computers using fit and loading the whole csv
# would be easier to work with,
# df = pd.read_csv(config.data_file_path, index_col=False)

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(None, 300, 49), activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(22, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

data_gen = PosTagDataGenerator(config.data_file_path, config.batch_size)

model.fit_generator(data_gen)
