

class WordAnalyse:
    def word_similar( word2vec_model, words): #词语相似度匹配
        similar_words = word2vec_model.wv.most_similar(words, topn=20)
        return similar_words
