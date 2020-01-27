from Generator import DataGenerator
from Model import BuildModel
import json
import pickle

if __name__ == "__main__":

    with open("processedData/meta.json") as fl:
        meta = json.load(fl)

    # parameter to test model
    test = True

    train_ids = range(1, 100)
    val_ids = range(100, 140)

    batch_size = 2
    if test == False:
        train_ids = range(1, 100000)
        val_ids = range(100000, 120000)
        batch_size = 14

    num_context_token = meta["max_context"]
    num_query_token = meta["max_question"]
    num_hidden_state = 100

    training_generator = DataGenerator(train_ids, batch_size=batch_size)
    validation_generator = DataGenerator(val_ids, batch_size=batch_size)

    model = BuildModel(num_context_token, num_query_token, num_hidden_state, meta["vocab_length"])


    # parameter to load and retrain model
    load_model = False
    if load_model == True:
      model.load_weights('weights.h5')
      model._make_train_function()
      with open('optimizer.pkl', 'rb') as f:
          weight_values = pickle.load(f)
      model.optimizer.set_weights(weight_values)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=5)
    model.save_weights('weights.h5')
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open('optimizer.pkl', 'wb') as f:
        pickle.dump(weight_values, f)