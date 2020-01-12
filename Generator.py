import numpy as np
import json
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        questions_array, context_array, ans_start_end_array = self.__getQuestionContext(list_IDs_temp)
        return [context_array, questions_array], [ans_start_end_array]

    def __getContext(self, context_id):
        context_file_id = int(context_id / 100) + 1 if context_id % 100 > 0 else int(context_id / 100)
        context_file = "processedData/context/" + str(context_file_id * 100) + ".json"
        with open(context_file) as fl:
            context = json.load(fl)[str(context_id)]
        return context
        

    def __getQuestionContext(self, list_IDs_temp):
        with open("processedData/meta.json") as fl:
            meta = json.load(fl)
        

        questions_array = []
        context_array = []
        ans_start_end_array = []

        start_index = min(list_IDs_temp)
        end_index = max(list_IDs_temp)
        min_index = int(min(list_IDs_temp) / 1000) + 1 if start_index % 1000 > 0 else int(start_index / 1000)
        max_index = int(max(list_IDs_temp) / 1000) + 1 if end_index % 1000 > 0 else int(end_index / 1000)

        current_context = -1
        context = []

        for i in range(min_index, max_index+1):
            question_file = "processedData/question/" + str(i * 1000) + ".json"
            with open(question_file) as fl:
                questions = json.load(fl)["data"]

            for j in range(start_index, i * 1000 + 1):
                question = questions[str(j)]
                if current_context != question["context"]:
                    current_context = question["context"]
                    context = self.__getContext(current_context)
                    context = context + [0] * (meta["max_context"] - len(context))
                q = question["question"]
                q = q + [0] * (meta["max_question"] - len(q))
                questions_array.append(q)
                context_array.append(context)
                ans_start_end_array.append([[question["start"], question["end"]]])

                if start_index == end_index:
                    break
                start_index = start_index + 1
        
        return np.array(questions_array), np.array(context_array), np.array(ans_start_end_array)