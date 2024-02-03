from gensim.models import Word2Vec
import numpy as np

class WordAnalyse:
    def word_similar( word2vec_model, words,sum): #词语相似度匹配
        similar_words = word2vec_model.wv.most_similar(words, topn=sum)
        return similar_words

    def inference_word(inferenceWord,inferencedWord,willInferenceWord,model):
        result = model.wv.most_similar(positive=[inferencedWord, willInferenceWord], negative=[inferenceWord], topn=6)
        return [item[0] for item in result]