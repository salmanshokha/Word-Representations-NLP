import os
import pickle
import numpy as np


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

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
def cos_similar(m,n):
    top = m.T.dot(n)
    bottom = np.linalg.norm(m)*np.linalg.norm(n)
    return 1.0*(top/bottom)

output_string = ""

# open file and read line by line
with open("word_analogy_test.txt") as file_in:
    newline = file_in.readline()

    # iterate each line
    while newline:
        words = newline.strip().split("||")[1]
        words_list = words.strip().split(',')
        similar = []
        for i in words_list:
            w1, w2 = i.strip().split(':')
            w1, w2 = w1[-1:], w2[:1]
            embed_1 = embeddings[dictionary[w1]]
            embed_2 = embeddings[dictionary[w2]]
            similar.append(cos_similar(embed_1, embed_2))

        smallest_index = similar.index(min(similar))
        largest_index = similar.index(max(similar))

        least_illustrative = words_list[smallest_index]
        most_illustrative = words_list[largest_index]

        output_string += words.strip().replace(","," ") + " " + least_illustrative + " " + most_illustrative + "\n"
        newline = file_in.readline()

# write to file
output_file = 'word_analogy_test_predictions_' + loss_model + '.txt'
file_out = open(output_file, "w")
file_out.write(output_string)
file_out.close()