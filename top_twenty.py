import os
import pickle
import numpy as np

model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'
model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))


def cos_similar(m, n):
    top = m.T.dot(n)
    bottom = np.linalg.norm(m) * np.linalg.norm(n)
    return 1.0 * (top / bottom)


words_list = ['first', 'american', 'would']
dict = {}

for word in words_list:
    similar_list = []
    dict[word] = []
    for word1, word2 in dictionary.items():
        if word1 != word:
            similarity = cos_similar(embeddings[word2], embeddings[dictionary[word]])
            similar_list.append((similarity, word1))

    similar_list.sort(key=lambda x: x[0], reverse=True)

    i = 0
    while i in range(0, 20):
        dict[word].append(similar_list[i][1])
        i += 1

print(dict)