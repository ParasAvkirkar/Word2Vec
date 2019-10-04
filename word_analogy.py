import os
import pickle
import numpy as np


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

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

        # print("Test cases read: {0}".format(test_cases))

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

    def get_most_least_illustration(self, test_cases, avg_vector):
        max_cossim = -1.0 # Default value beyond minimum possible value of cosine-similarity
        min_cossim = 2.0 # Default value beyond maximum possible value of cosine-similarity
        most_ill = None
        least_ill = None
        for pair in test_cases:
            key_embedding = embeddings[dictionary[pair[0]]]
            value_embedding = embeddings[dictionary[pair[1]]]
            diff_vec = self.get_diff_vector(key_embedding, value_embedding)
            cossim = self.get_cossim(diff_vec, avg_vector)

            if max_cossim < cossim:
                max_cossim = cossim
                most_ill = pair
            if min_cossim > cossim:
                min_cossim = cossim
                least_ill = pair

        return (most_ill, least_ill)

    def get_cossim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def execute(self, input_file_name, output_file_name):
        all_test_cases = self.read_test_cases(input_file_name)
        dev_predictions = []
        for case in all_test_cases:
            avg_vector = self.get_avg_diff_vector(case['truth'])
            most_ill, least_ill = self.get_most_least_illustration(case['test_case'], avg_vector)
            case['illus'] = (most_ill, least_ill)
            dev_predictions.append(case)

        self.write_predictions(dev_predictions, output_file_name)

    def write_predictions(self, dev_predictions, output_file_name):
        with open(output_file_name, 'w') as f:
            for case in dev_predictions:
                line = ["\"" + p[0] + ":" + p[1] + "\"" for p in case['test_case']]
                illustrations = case['illus']
                line.append("\"" + illustrations[0][0] + ":" + illustrations[0][1] + "\"")
                line.append("\"" + illustrations[1][0] + ":" + illustrations[1][1] + "\"")
                f.write(" ".join(line) + "\n")

if __name__ == "__main__":
    word_analogy_task = Word_Analogy_Task()
    word_analogy_task.execute("word_analogy_dev.txt", "nce_predictions_first.txt")
