import math

import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer


class PosTagDataGenerator:

    def __init__(self, file_name, batch_size, steps_per_epoch=None):
        super(PosTagDataGenerator, self).__init__(file_name, batch_size, steps_per_epoch)
        self.batch_size = batch_size
        self.file_name = file_name
        self.seq_length = PosTagDataGenerator.calc_num_of_rows(self.file_name, include_header=True)
        if steps_per_epoch is None:
            self.steps_per_epoch = self.__len__()
        self.step = 0

        self.gen = self.create_csv_gen()

        self.pos_encoder = PosIntegerEncoding(50)
        # TODO: Create country encoder
        self.country_encoder = None

    def create_csv_gen(self):
        return pd.read_csv(self.file_name, chunksize=self.batch_size)

    def __len__(self):
        return math.floor(self.seq_length / self.batch_size)

    def on_epoch_end(self):
        self.gen.close()
        self.gen = self.create_csv_gen()

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.steps_per_epoch:
            self.step = 0
            self.on_epoch_end()
        self.step += 1
        return self.process_rows()

    def process_rows(self):
        df = self.gen.get_chunk()
        x = df.text.apply(lambda s: to_categorical(self.pos_encoder.encode(s), num_classes=49))
        # TODO: Check for the number of countries/languages
        y = df.country.apply(lambda c: to_categorical(self.country_encoder.enocode(c), num_classes=None))
        return x, y

    @staticmethod
    def calc_num_of_rows(file_name, include_header=None):
        with open(file_name, 'r') as f:
            s = sum(1 for _ in f)

        if not include_header:
            s -= 1

        return s


class PosIntegerEncoding:

    def __init__(self, num_words=None) -> None:
        super().__init__()
        self._max = 0
        self._encoding = {}
        self._tokenizer = Tokenizer(num_words=num_words, filters='')

    def encode(self, text):
        vec = []
        # TODO: Check if tokenizer works properly on PoS tags
        seq = self._tokenizer.texts_to_sequences(text)
        for token in seq:
            if token not in self._encoding:
                self._max += 1
                self._encoding[token] = self._max

            vec.append(self._encoding.get(token))

        return vec
