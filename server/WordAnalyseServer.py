

class WordAnalyse:
    def word_similar( word2vec_model, words,sum): #词语相似度匹配
        similar_words = word2vec_model.wv.most_similar(words, topn=sum)
        return similar_words

    def word_inferenceWord(word2Vec_model,inference_word,inferenced_word,will_inference_word):
        inference_word = word2Vec_model.wv.most_similar(positive=[inferenced_word, will_inference_word], negative=[inference_word], topn=6)
        return [item[0] for item in inference_word]