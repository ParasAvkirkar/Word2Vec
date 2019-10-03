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
    # Citation: https://datascience.stackexchange.com/questions/15032/how-to-do-batch-inner-product-in-tensorflow
    u0vC = tf.reduce_sum(tf.multiply(true_w, inputs), axis=1)
    logEU0VC = tf.log(tf.exp(u0vC))

    each_input_product = tf.matmul(inputs, true_w, transpose_b=True)
    each_input_product = tf.exp(each_input_product)
    each_input_product = tf.reduce_sum(each_input_product, axis=1)
    logDen = tf.log(each_input_product)


    A = logEU0VC
    B = tf.reshape(logDen, [inputs.get_shape().as_list()[0], 1])

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
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
