import os
import pickle
import numpy as np
from model_params import ModelParams
from word2vec_basic import get_model_name


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'
model_params = ModelParams()
print("Testing on: " + str(model_params))
model_filepath = os.path.join(model_path, get_model_name(batch_size=model_params.batch_size, skip_window=model_params.skip_window,
                                                             num_skips=model_params.num_skips, num_sampled=model_params.num_sampled,
                                                             max_num_steps=model_params.max_num_steps, learning_rate=model_params.learning_rate,
                                                             loss_model=loss_model) + ".model")
print(model_filepath)
print("generating predictions based on: " + model_filepath)
# word2vec_64_4_8_64_200001_1.0_cross_entropy.model

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
embedding_dim = embeddings[1].shape[0]


"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

class Word_Analogy_Task:

    def read_test_cases(self, file_name):
        print("Reading test-cases")
        test_cases = []
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                answer_line = line.split("||")[0]
                question_line = line.split("||")[1]
                true_related = self.parse_pairs(answer_line)
                test_case = self.parse_pairs(question_line)

                test_cases.append({"truth": true_related, "test_case": test_case})

        print("Test cases read: {0}".format(str(len(test_cases))))

        return test_cases

    def parse_pairs(self, line):
        pairs = []
        cols = line.split(",")
        for pair_line in cols:
            pair_line = pair_line.strip("\"")
            pairs.append((pair_line.split(":")[0], pair_line.split(":")[1]))

        return pairs

    def get_diff_vector(self, v1, v2):
        return np.subtract(v2, v1)

    def get_avg_diff_vector(self, related_pairs):
        avg_vector = np.zeros(embedding_dim) # [0.0, 0.0, .... 0.0]
        for pair in related_pairs:
            key_embedding = embeddings[dictionary[pair[0]]]
            value_embedding = embeddings[dictionary[pair[1]]]
            diff_vec = self.get_diff_vector(key_embedding, value_embedding)
            avg_vector = np.add(avg_vector, diff_vec)

        avg_vector = np.true_divide(avg_vector, len(related_pairs))

        return avg_vector

    def get_avg_diff_dist(self, related_pairs):
        avg_dist = 0.0
        for pair in related_pairs:
            key_embedding = embeddings[dictionary[pair[0]]]
            value_embedding = embeddings[dictionary[pair[1]]]
            diff_vec = self.get_diff_vector(key_embedding, value_embedding)
            avg_dist += np.linalg.norm(diff_vec)

        avg_dist = avg_dist/len(related_pairs)

        return avg_dist

    def get_avg_cossim(self, related_pairs):
        avg_cossim = 0.0
        for pair in related_pairs:
            key_embedding = embeddings[dictionary[pair[0]]]
            value_embedding = embeddings[dictionary[pair[1]]]
            avg_cossim += self.get_cossim(key_embedding, value_embedding)

        avg_cossim = avg_cossim/len(related_pairs)

        return avg_cossim

    def get_most_least_illustration(self, test_cases, avg_vector, avg_dist, avg_cossim):
        max_score = -float("infinity") # Default value beyond minimum possible value of cosine-similarity
        min_score = float("infinity") # Default value beyond maximum possible value of cosine-similarity

        most_ill = None
        least_ill = None
        for pair in test_cases:
            key_embedding = embeddings[dictionary[pair[0]]]
            value_embedding = embeddings[dictionary[pair[1]]]
            diff_vec = self.get_diff_vector(key_embedding, value_embedding)
            diff_cossim = self.get_cossim(diff_vec, avg_vector)
            pairwise_cossim = self.get_cossim(key_embedding, value_embedding)
            absolute_euclid_dist_diff = self.get_absolute_euclid_dist_diff(diff_vec, avg_vector)
            dist_diff = abs(np.linalg.norm(diff_vec) - avg_dist)
            score = self.get_score(pairwise_cossim, diff_cossim, absolute_euclid_dist_diff, dist_diff, avg_cossim)
            if max_score < score:
                max_score = score
                most_ill = pair
            if min_score > score:
                min_score = score
                least_ill = pair
        return (most_ill, least_ill)

    def get_score(self, pairwise_cossim, diff_cossim, absolute_euclid_dist_diff, dist_diff, avg_cossim):
        return 1.0/(abs(avg_cossim - pairwise_cossim) * dist_diff)

    def get_absolute_euclid_dist_diff(self, v1, v2):
        v1_l2_norm = np.linalg.norm(v1)
        v2_l2_norm = np.linalg.norm(v2)
        return abs(v1_l2_norm - v2_l2_norm)

    def get_cossim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def execute(self, input_file_name, output_file_name):
        all_test_cases = self.read_test_cases(input_file_name)
        dev_predictions = []
        print("execute::Calculating most-least illustrative pairs")
        for case in all_test_cases:
            avg_vector = self.get_avg_diff_vector(case['truth'])
            avg_dist = self.get_avg_diff_dist(case['truth'])
            avg_cossim = self.get_avg_cossim(case['truth'])
            most_ill, least_ill = self.get_most_least_illustration(case['test_case'], avg_vector, avg_dist, avg_cossim)
            case['illus'] = {'most': most_ill, 'least': least_ill}
            dev_predictions.append(case)
        print("execute::Done calculation: {0}".format(str(len(dev_predictions))))
        self.write_predictions(dev_predictions, output_file_name)

    def write_predictions(self, dev_predictions, output_file_name):
        print("write_predictions::Begin::{0}::{1}".format(str(len(dev_predictions)), output_file_name))
        with open(output_file_name, 'w') as f:
            for case in dev_predictions:
                line = ["\"" + p[0] + ":" + p[1] + "\"" for p in case['test_case']]
                illustrations = case['illus']
                line.append("\"" + illustrations['least'][0] + ":" + illustrations['least'][1] + "\"")
                line.append("\"" + illustrations['most'][0] + ":" + illustrations['most'][1] + "\"")
                f.write(" ".join(line) + "\n")

        print("write_predictions::Done::{0}::{1}".format(str(len(dev_predictions)), output_file_name))

if __name__ == "__main__":
    word_analogy_task = Word_Analogy_Task()
    batch_size, skip_window, num_skips, num_sampled, max_num_steps, learning_rate = model_params.get_tuning_params()
    model_name = get_model_name(batch_size=batch_size, skip_window=skip_window,
                   num_skips=num_skips, num_sampled=num_sampled,
                   max_num_steps=max_num_steps, learning_rate=learning_rate,
                   loss_model=loss_model)
    result_file_name = model_name + "_results.txt"
    stats_file_name = model_name + "_stats.txt"
    word_analogy_task.execute("word_analogy_dev.txt", result_file_name)
    cmd = "perl score_maxdiff.pl word_analogy_dev_mturk_answers.txt " + result_file_name + " " + stats_file_name
    print(os.system(cmd))
    print("cat " + stats_file_name)
    # print(os.system("cat " + stats_file_name))

# word2vec_128_4_8_64_200001_1.0_cross_entropy.model

# perl score_maxdiff.pl word_analogy_dev_mturk_answers.txt nce_predictions_first.txt nce_turk_results.txt
#  perl score_maxdiff.pl word_analogy_dev_sample_predictions.txt word2vec_128_4_8_64_200001_1.0_nce_results.txt word2vec_128_4_8_64_200001_1.0_nce_stats.txt
# perl score_maxdiff.pl word_analogy_dev_mturk_answers.txt word2vec_128_4_8_64_200001_1.0_nce_results.txt word2vec_128_4_8_64_200001_1.0_nce_stats.txt