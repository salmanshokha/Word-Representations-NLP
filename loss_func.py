import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================
    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
    Write the code that calculate A = log(exp({u_o}^T v_c))
    A =
    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})
    B =
    ==========================================================================
    """
    
    mat = tf.matmul(inputs, true_w, transpose_b=True)
    A = tf.diag_part(mat)  # A = log(exp({true_w}^T inputs))
    B = tf.log(tf.reduce_sum(tf.exp(mat), axis=1))  # B = log(\sum{exp({true_w}^T inputs)})
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
    import numpy as np

    sample_size = len(sample)
    input_size = inputs.get_shape().as_list()
    batch_size, embedding_size = input_size[0], input_size[1]

    unigram_prob_tensor = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    pr_pd = tf.reshape(tf.nn.embedding_lookup(unigram_prob_tensor, labels), [batch_size])
    pr_pn = np.ndarray(shape=(sample_size), dtype=np.float32)

    counter = 0
    while counter < sample_size:
        uni_prob = unigram_prob[sample[counter]]
        pr_pn[counter] = uni_prob
        counter += 1

    t1 = tf.reshape(tf.nn.embedding_lookup(weights, labels), [-1, embedding_size])
    t2 = tf.nn.embedding_lookup(weights, sample)

    tensor1 = tf.diag_part(tf.matmul(inputs, tf.transpose(t1)))
    tensor2 = tf.matmul(inputs, tf.transpose(t2))

    bias_labels = tf.reshape(tf.nn.embedding_lookup(biases, labels), [batch_size])
    bias_sample = tf.nn.embedding_lookup(biases, sample)

    term1 = tf.log(tf.add(sample_size * pr_pd, 1e-10))
    term1 = tf.subtract(tensor1 + bias_labels, term1)
    term1 = tf.sigmoid(term1)

    term2 = tf.log(tf.add(sample_size * pr_pn, 1e-10))
    term2 = tf.subtract(tensor2 + bias_sample, term2)
    term2 = tf.sigmoid(term2)

    A = tf.log(term1 + 1e-10)
    B = tf.reduce_sum(tf.log(1.0 - term2 + 1e-10),1)

    nce_loss_result = tf.negative(tf.add(A, B))
    return nce_loss_result

    """
    Reading Sources:
    # https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
    # https://arxiv.org/pdf/1410.8251.pdf
    # https://www.quora.com/What-is-Noise-Contrastive-estimation-NCE
    # tf.reduce_sum: https://stackoverflow.com/questions/47157692/how-does-reduce-sum-work-in-tensorflow
    # tf.negative : Computes numerical negative value element-wise.
    """