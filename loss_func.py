import tensorflow as tf
import numpy as np


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
    # return tf.subtract(B, A)

    # A = log(exp({u_o} ^ T v_c))
    a = tf.log(tf.exp(tf.matmul(true_w, inputs, transpose_a=True, transpose_b=False))+1e-10)
    # B = log(\sum{exp({u_w} ^ T v_c)})
    b = tf.log(tf.reduce_sum(tf.exp(tf.matmul(true_w, inputs, transpose_a=True, transpose_b=False)),axis=1)+1e-10)
    return tf.subtract(b, a)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):

    """
    :param inputs: batch_size x embedding_size
    :param weights: Vocabulary x embedding_size
    :param biases: Vocabulary x 1
    :param labels: batch_size,1
    :param sample: num_sampled
    :param unigram_prob: Vocabulary
    :return:
    """
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embedding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimension is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimension is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    
    #
    # Direct implementation of the formula from the paper
    # "Learning word embeddings efficiently with noise-contrastive estimation"
    # Refer this link: https://www.cs.toronto.edu/~amnih/papers/wordreps.pdf
    
    embedding_size = weights.shape[1]               #As weights has dimensions[vocab, embedding_size]
    num_sampled = sample.shape[0]                   #sample has dimensions[num_sampled]

    Uo = tf.nn.embedding_lookup(weights, labels)   # Word embeddings for predicting words
    Uo = tf.reshape(Uo, [-1, embedding_size])      #Reshaping according to embedding size, the -1 is used to infer the shape
                                                   #Referred tensorflow doc for reshape
                                                   # https://www.tensorflow.org/api_docs/python/tf/manip/reshape
    Uo_Transpose = tf.transpose(Uo)
    Uo_Transpose_Uc = tf.matmul(inputs, Uo_Transpose)   #Multiplication
   
    Qw = tf.nn.embedding_lookup(weights, sample)   #Word embeddings for negative samples
    Qw_Transpose = tf.transpose(Qw)
    Qw_Transpose_Uc = tf.matmul(inputs, Qw_Transpose)   #Multiplication

    # Create a vector of unigram probability since @unigram_prob is a vector
    unigram_probability_tensor = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    negative_bias = tf.nn.embedding_lookup(biases, sample)   # Getting the bias for negative samples
                                                        # Here sample is the 2nd parameter as NCE is based on k-negative sampling
    negative_bias = tf.reshape(negative_bias, [-1])               #Reshaping the bias vector
    negative_score = tf.nn.bias_add(Qw_Transpose_Uc, negative_bias)
    
    positive_bias = tf.nn.embedding_lookup(biases, labels)          #Extracting the bias for positive samples
    positive_bias = tf.reshape(positive_bias, [-1])                 #Reshaping since @labels has an extra axis
    positive_score = tf.nn.bias_add(Uo_Transpose_Uc, positive_bias) #getting positive score from positive bias

    negative_unigram_prob = tf.nn.embedding_lookup(unigram_probability_tensor, sample)       #Getting probability of noise
    positive_unigram_prob = tf.nn.embedding_lookup(unigram_probability_tensor, labels)  #Getting positive unigram probability
    positive_unigram_prob = tf.reshape(positive_unigram_prob, [-1])                     #Reshaping because of @labels

    # We have two sigmoid terms, first which is a positive, and second which is a negative

    # sigmoid(wo, wc) +/- log(k * positive/negative unigram probabilities)
    sigmoid1 = tf.sigmoid(tf.subtract(positive_score, tf.log(0.000000001
                                                       + tf.scalar_mul(num_sampled, positive_unigram_prob))))
    
    sigmoid2 = tf.sigmoid(tf.subtract(negative_score, tf.log(0.00000001
                                                       + tf.scalar_mul(num_sampled, negative_unigram_prob))))
    # Correction value of 1e-10 has been added to both the sigmoids in order to avoid errors due to NaN values.

    a = tf.log(sigmoid1 + 1e-10)                               #Calculating positive probability
    b = tf.reduce_sum(tf.log(1 - sigmoid2 + 0.0000000001), 1)  #Calculate 1+ negative of sigmoid2 i.e. -ve probability
                                                                #take it's log and take its sum over -ve V^k

    return tf.scalar_mul(-1, tf.add(a, b))