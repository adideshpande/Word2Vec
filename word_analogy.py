import os
import pickle
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine as cosine_similarity

model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce_try3'
model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
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
examples = []
choices = []
average_cosine = []
pred_file = []
output_file = open("word_analogy_test_predictions_"+loss_model+"_bestmodel.txt", "w")

with open('./word_analogy_test.txt') as f:
    for every_line in f:
        every_line = every_line.strip()
        two_halves = every_line.split("||")
        examples.append(two_halves[0]), choices.append(two_halves[1])

for s, t in zip(examples, choices):
    max_distance, min_distance = np.NINF, np.Infinity
    average_train_cosine = []
    cosine_element_test = []

    tupleset = s.split(",")
    tupleset = [a.strip('\"') for a in tupleset]
    for tuple in tupleset:
        key, value = tuple.split(":")
        average_train_cosine.append(cosine_similarity(embeddings[dictionary[key]],
                                                         embeddings[dictionary[value]]))

    quadrupleset = t.split(",")
    quadrupleset = [a.strip('\"') for a in quadrupleset]
    for tuple_2 in quadrupleset:
        key_test, value_test = tuple_2.split(":")
        cosine_element_test.append(cosine_similarity(embeddings[dictionary[key_test]],
                                                embeddings[dictionary[value_test]]))
    
    average_cosine = sum(average_train_cosine) / len(average_train_cosine)
    for idx, distance in enumerate(cosine_element_test):
        if distance > max_distance:
            max_dist, max_idx = distance, idx
        if distance < min_distance:
            min_dist, min_idx = distance, idx
    pred_file.append([quadrupleset, quadrupleset[min_idx], quadrupleset[max_idx]])

for x in pred_file:
    final_string = []
    final_string.append('\" \"'.join(first for first in x[0])), final_string.append('\" \"'.join(next_three for
                                                                                                 next_three in x[1:]))
    for last_two in final_string:
        output_file.write('\"' + last_two + '\" ')
    output_file.write('\n')
output_file.close()