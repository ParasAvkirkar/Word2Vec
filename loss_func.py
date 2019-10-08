import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A = N


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    input_shape = inputs.get_shape().as_list()
    n_batch = input_shape[0]
    # Citation: https://datascience.stackexchange.com/questions/15032/how-to-do-batch-inner-product-in-tensorflow
    u0vC = tf.reduce_sum(tf.multiply(true_w, inputs), axis=1)
    epsilon_vec = get_epsilon_vector(n_batch, 0.0000000001)
    logEU0VC = tf.log(tf.add(tf.exp(u0vC), epsilon_vec))

    each_input_product = tf.matmul(inputs, true_w, transpose_b=True)
    each_input_product = tf.exp(each_input_product)
    each_input_product = tf.reduce_sum(each_input_product, axis=1)
    each_input_product = tf.reshape(each_input_product, [n_batch, 1])
    logDen = tf.log(tf.add(each_input_product, epsilon_vec))

    A = logEU0VC
    B = tf.reshape(logDen, [n_batch, 1])

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, samples, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    input_shape = inputs.get_shape().as_list()
    # print(str(input_shape))
    n_batch = input_shape[0]
    n_embedding = input_shape[1]
    n_samples = len(samples)
    # print("n_batch: {0}, n_embedding: {1}, n_samples: {2}".format(str(n_batch), str(n_embedding), str(n_samples)))
    # inputs = tf.Print(inputs, [inputs], "inputs", summarize=32)

    true_prob = calculate_true_distribution_prob(inputs, weights, biases, labels, unigram_prob, n_batch, n_samples, n_embedding) # batch x 1

    noise_prob = calculate_noise_distribution_prob(inputs, weights, biases, samples, unigram_prob, n_batch, n_samples, n_embedding) # batch x n
    noise_prob = tf.reduce_sum(noise_prob, axis=1)

    nce = tf.add(true_prob, noise_prob)
    nce = tf.scalar_mul(-1, nce)

    return nce


def calculate_true_distribution_prob(inputs, weights, biases, labels, unigram_prob, n_batch, n_samples, n_embedding):
    positive_sample_score = get_positive_sample_score(inputs, weights, biases, labels, n_batch, n_embedding)

    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    predicted_unigram_prob = tf.reshape(tf.nn.embedding_lookup(unigram_prob, labels), [n_batch, 1])
    predicted_unigram_prob = tf.scalar_mul(n_samples, predicted_unigram_prob)
    epsilon_vec = get_epsilon_vector(n_batch, 0.0000000001)
    predicted_unigram_prob = tf.log(tf.add(predicted_unigram_prob, epsilon_vec))
    # print("predicted_unigram_prob shape: " + str(predicted_unigram_prob.get_shape()))
    final_true_prob = tf.subtract(positive_sample_score, predicted_unigram_prob)
    final_true_prob = tf.sigmoid(final_true_prob)
    final_true_prob = tf.log(tf.add(final_true_prob, epsilon_vec))
    # print("final_true_prob shape: " + str(final_true_prob.get_shape()))

    return final_true_prob


def calculate_noise_distribution_prob(inputs, weights, biases, samples, unigram_prob, n_batch, n_samples, n_embedding):
    negative_sample_score = get_negative_sample_score(inputs, weights, biases, samples, n_samples, n_batch, n_embedding)

    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    negative_unigram_prob = tf.reshape(tf.nn.embedding_lookup(unigram_prob, samples), [n_samples, 1])
    negative_unigram_prob = tf.transpose(negative_unigram_prob)
    negative_unigram_prob = tf.scalar_mul(n_samples, negative_unigram_prob)
    epsilon_vec = get_epsilon_vector(n_samples, 0.0000000001)
    epsilon_vec = tf.transpose(epsilon_vec)
    negative_unigram_prob = tf.log(tf.add(negative_unigram_prob, epsilon_vec))
    negative_unigram_prob = tf.tile(negative_unigram_prob, [n_batch, 1])
    # print("negative_unigram_prob shape: " + str(negative_unigram_prob.get_shape()))

    ones = tf.ones([n_batch, n_samples], tf.float32)
    # print("ones shape: " + str(ones.get_shape()))
    final_noise_prob = tf.subtract(negative_sample_score, negative_unigram_prob)
    final_noise_prob = tf.sigmoid(final_noise_prob)
    final_noise_prob = tf.subtract(ones, final_noise_prob)\

    epsilon_mat = get_epsilon_matrix(n_batch, n_samples, 0.0000000001)
    final_noise_prob = tf.log(tf.add(final_noise_prob, epsilon_mat))
    # print("final_noise_prob shape: " + str(final_noise_prob.get_shape()))

    return final_noise_prob

# s(wx, wc) = (uxTvc) + bx
# Returns shape batch x k
def get_negative_sample_score(inputs, weights, biases, samples, n_samples, n_batch, n_embedding):
    negative_embeddings = tf.nn.embedding_lookup(weights, samples)
    # print("negative_embeddings shape: " + str(negative_embeddings.get_shape()))
    input_negative_product = tf.matmul(inputs, negative_embeddings, transpose_b=True)
    # print("input_negative_product shape: " + str(input_negative_product.get_shape()))
    negative_samples_bias = tf.reshape(tf.nn.embedding_lookup(biases, samples), [n_samples, 1])
    negative_samples_bias = tf.transpose(negative_samples_bias)
    # print("negative_samples_bias shape: " + str(negative_samples_bias.get_shape()))
    negative_samples_bias = tf.tile(negative_samples_bias, [n_batch, 1])
    # print("negative_samples_bias shape: " + str(negative_samples_bias.get_shape()))
    negative_sample_score = tf.add(input_negative_product, negative_samples_bias)
    # print("negative_sample_score shape: " + str(negative_sample_score.get_shape()))
    # Returns tensor of dim: batch x k
    return negative_sample_score


# s(w0, wc) = (uoTvc) + b0
# Return shape batch x 1
def get_positive_sample_score(inputs, weights, biases, labels, n_batch, n_embedding):
    u_embeddings = tf.reshape(tf.nn.embedding_lookup(weights, labels), [n_batch, n_embedding])
    # print("u_embeddings shape: " + str(u_embeddings.get_shape()))
    positive_sample_score = tf.reshape(tf.reduce_sum(tf.multiply(u_embeddings, inputs), axis=1), [n_batch, 1])
    # print("positive_sample_score shape: " + str(positive_sample_score.get_shape()))
    predicted_labels_bias = tf.nn.embedding_lookup(biases, labels)
    # print("predicted_labels_bias shape: " + str(predicted_labels_bias.get_shape()))
    positive_sample_score = tf.add(positive_sample_score, predicted_labels_bias)
    # print("positive_sample_score shape: " + str(positive_sample_score.get_shape()))
    return positive_sample_score

def get_epsilon_matrix(n, m, epsilon):
    mat = []
    for i in range(n):
        mat.append([epsilon for j in range(m)])
    tensor = tf.reshape(tf.convert_to_tensor(mat, dtype=tf.float32), [n, m])

    return tensor

def get_epsilon_vector(n, epsilon):
    vec = [epsilon for i in range(n)]
    tensor = tf.reshape(tf.convert_to_tensor(vec, dtype=tf.float32), [n, 1])

    return tensor