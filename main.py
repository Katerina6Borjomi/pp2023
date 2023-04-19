#import gensim
import numpy as np

# Load pre-trained word2vec model
#import gensim.downloader as api

#model = api.load("word2vec-ruscorpora-300")
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/borjomy/PycharmProjects/pp/glove1/GoogleNews-vectors-negative300.bin', binary=True)

# Define function to calculate similarity between two texts
#def text_similarity(text1, text2):
    # Split texts into list of words
#    words1 = text1.split()
#    words2 = text2.split()

    # Remove words not in the model's vocabulary
#    words1 = [word for word in words1 if word in model.key_to_index]
#    words2 = [word for word in words2 if word in model.key_to_index]

    # Calculate the similarity between the texts using word2vec
#    if len(words1) > 0 and len(words2) > 0:
#        word_vectors1 = np.array([model.get_vector(word) for word in words1])
#        word_vectors2 = np.array([model.get_vector(word) for word in words2])
#        similarity_matrix = word_vectors1 @ word_vectors2.T
#        similarity = np.mean(similarity_matrix)
#    else:
#        similarity = 0

#    return similarity

# Example usage
#text1 = "На окраине нашего городка распологался жутковатый район, именованный в народе Стрельбищем. Раньше тут армейская часть была. После часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. заполучила тут участок и наша тетушка. И надо сказать, безгранично этому обрадовалась. Земли-то у нас богатые — что желаешь сажай, все прорастет. Стала она на свою дачу приезжать по выходным. Да вот беда, объявилась в тех краях собака бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — сердце в пятки уйдет, а уж если погонится — тут и окончание сказочке."
#text2 = "На окраине нашего города находился диковатый район, называемый в народе Стрельбищем. Раньше здесь воинская часть была. Потом часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. Получила здесь участок и наша тетушка. И надо сказать, очень этому обрадовалась. Земли-то у нас богатые — что хочешь сажай, все прорастет. Стала она на свою дачу ездить по выходным. Да вот беда, объявилась в тех краях собачонка бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — душа в пятки уйдет, а уж если погонится — тут и конец сказочке."

#print(text_similarity(text1, text2))

#====================================================================

#import gensim.downloader as api

# Load the pre-trained FastText model for Russian
#model = api.load("fasttext-wiki-news-subwords-300")

# Define function to calculate similarity between two texts
#def text_similarity(text1, text2):
    # Split texts into list of words
#    words1 = text1.split()
#    words2 = text2.split()

    # Remove words not in the model's vocabulary
#    words1 = [word for word in words1 if word in model.key_to_index]
#    words2 = [word for word in words2 if word in model.key_to_index]

    # Calculate the similarity between the texts using FastText
#    if len(words1) > 0 and len(words2) > 0:
#        word_vectors1 = np.array([model[word] for word in words1])
#        word_vectors2 = np.array([model[word] for word in words2])
#        similarity_matrix = word_vectors1 @ word_vectors2.T
#        similarity = np.mean(similarity_matrix)
#    else:
#        similarity = 0

#    return similarity

# Example usage
#text1 = "На окраине нашего городка распологался жутковатый район, именованный в народе Стрельбищем. Раньше тут армейская часть была. После часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. заполучила тут участок и наша тетушка. И надо сказать, безгранично этому обрадовалась. Земли-то у нас богатые — что желаешь сажай, все прорастет. Стала она на свою дачу приезжать по выходным. Да вот беда, объявилась в тех краях собака бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — сердце в пятки уйдет, а уж если погонится — тут и окончание сказочке."
#text2 = "На окраине нашего города находился диковатый район, называемый в народе Стрельбищем. Раньше здесь воинская часть была. Потом часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. Получила здесь участок и наша тетушка. И надо сказать, очень этому обрадовалась. Земли-то у нас богатые — что хочешь сажай, все прорастет. Стала она на свою дачу ездить по выходным. Да вот беда, объявилась в тех краях собачонка бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — душа в пятки уйдет, а уж если погонится — тут и конец сказочке."

#text1 = "Вчера вечером я была в магазине"
#text2 = "Вчера ночью мы были в магазине"

#print(text_similarity(text1, text2))

#============================================================

import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained FastText model for Russian
model = api.load("fasttext-wiki-news-subwords-300")

# Preprocess the texts by removing words not in the model's vocabulary
def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word in model.key_to_index]
    return ' '.join(words)

# Define function to calculate similarity between two texts
def text_similarity(text1, text2):
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Calculate the embeddings for the preprocessed texts using FastText
    embeddings1 = np.array([model[word] for word in text1.split()])
    embeddings2 = np.array([model[word] for word in text2.split()])

    # Calculate the similarity between the embeddings using cosine similarity
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    similarity = np.mean(similarity_matrix)

    return similarity

# Example usage
text1 = "На окраине нашего городка распологался жутковатый район, именованный в народе Стрельбищем. Раньше тут армейская часть была. После часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. заполучила тут участок и наша тетушка. И надо сказать, безгранично этому обрадовалась. Земли-то у нас богатые — что желаешь сажай, все прорастет. Стала она на свою дачу приезжать по выходным. Да вот беда, объявилась в тех краях собака бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — сердце в пятки уйдет, а уж если погонится — тут и окончание сказочке."
text2 = "На окраине нашего города находился диковатый район, называемый в народе Стрельбищем. Раньше здесь воинская часть была. Потом часть закрыли, пустыри-полигоны репьем поросли и их стали под дачи раздавать. Получила здесь участок и наша тетушка. И надо сказать, очень этому обрадовалась. Земли-то у нас богатые — что хочешь сажай, все прорастет. Стала она на свою дачу ездить по выходным. Да вот беда, объявилась в тех краях собачонка бешеная. Даже не собачонка, а целое собачище: роста великаньего, зубищи во все стороны. Как зарычит — душа в пятки уйдет, а уж если погонится — тут и конец сказочке."

print(text_similarity(text1, text2))