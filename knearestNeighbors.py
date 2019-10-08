import os
import pickle
import numpy as np
import operator
from model_params import ModelParams
from word2vec_basic import get_model_name


model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'
model_params = ModelParams()
print("Testing on: " + str(model_params))
model_filepath = os.path.join(model_path, get_model_name(batch_size=model_params.batch_size, skip_window=model_params.skip_window,
                                                             num_skips=model_params.num_skips, num_sampled=model_params.num_sampled,
                                                             max_num_steps=model_params.max_num_steps, learning_rate=model_params.learning_rate,
                                                             loss_model=loss_model) + ".model")
print(model_filepath)
print("generating predictions based on: " + model_filepath)

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
embedding_dim = embeddings[1].shape[0]

def get_top_k_nearest(word, k):
    v_c = embeddings[dictionary[word]]
    score_dic = {}
    for query_word in dictionary:
        u_o = embeddings[dictionary[query_word]]
        score_dic[query_word] = np.dot(u_o, v_c)

    nearest_neighbors = sorted(score_dic.items(), key=operator.itemgetter(1), reverse=True)

    return nearest_neighbors[:k]

if __name__ == "__main__":
    batch_size, skip_window, num_skips, num_sampled, max_num_steps, learning_rate = model_params.get_tuning_params()
    model_name = get_model_name(batch_size=batch_size, skip_window=skip_window,
                   num_skips=num_skips, num_sampled=num_sampled,
                   max_num_steps=max_num_steps, learning_rate=learning_rate,
                   loss_model=loss_model)

    for word in ['first', 'american', 'would']:
        knn = get_top_k_nearest(word, 21)
        output = word + "-> "
        output = output + ", ".join(neighbour[0] for neighbour in knn)
        print(output)
        print(str(knn))

