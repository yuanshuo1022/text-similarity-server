from flask import Flask, request, jsonify, make_response, Blueprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from server.TextSimServer import SimilarityServer
from server.WordAnalyseServer import WordAnalyse
import gensim
from gensim.models.word2vec import PathLineSentences

import os

similarity_route = Blueprint('similarity_route', __name__)

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# 构建模型文件的绝对路径
model_path = os.path.join(parent_dir, 'model', 'one.model')
# 加载 Word2Vec 模型
model = gensim.models.Word2Vec.load(model_path)


@similarity_route.route('/api/analyze-similarity', methods=['POST'])
def calculate_similarity():
    try:
        # algorithm = request.form['algorithm']
        data = request.get_json()
        text_vector = int(data.get('textVector'))
        text1 = data.get('text1')
        text2 = data.get('text2')

        if text_vector == 1:
            # 平均向量计算
            vec1 = SimilarityServer.text_to_average_vector(text1, model)
            vec2 = SimilarityServer.text_to_average_vector(text2, model)
        else:
            # 使用 TF-IDF 向量化文本
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectorizer.fit([text1, text2])

            # 获取单词到 TF-IDF 权重的映射
            word2tfidf = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
            vec1 = SimilarityServer.sentence_vector(text1, model, word2tfidf)
            vec2 = SimilarityServer.sentence_vector(text2, model, word2tfidf)
            print(vec1)

        cosine_sim = cosine_similarity([vec1], [vec2])[0][0]
        correlation_sim = SimilarityServer.correlation_similarity(vec1, vec2)

        # 返回 JSON 数据
        return jsonify({'code': 200, 'cosine_sim': float(cosine_sim), 'correlation_sim': float(correlation_sim)})
    except Exception as e:
        # 处理异常情况
        return jsonify({'code': 300, 'error': str(e)})


@similarity_route.route('/api/word-similarity', methods=['POST'])
def word_similarity():
    try:
        data = request.get_json()
        words = data.get('singleWord')
        sum = data.get('sum')
        similar_word = WordAnalyse.word_similar(model, words, sum)
        return jsonify({'code': 200, 'similar_word': similar_word})
    except Exception as e:
        return jsonify({'code': 300, 'error': str(e)})


@similarity_route.route('/api/inferenceWord', methods=['POST'])
def infer_word():
    try:
        data = request.get_json()
        inference_word = data.get('inferenceWord')
        inferenced_word = data.get('inferencedWord')
        will_inference_word=data.get('willInferenceWord')
        inference_words = WordAnalyse.word_inferenceWord(model, inference_word,inferenced_word,will_inference_word)
        return jsonify({'code': 200, 'inference_words': inference_words})
    except Exception as e:
        return jsonify({'code': 300, 'error': str(e)})
