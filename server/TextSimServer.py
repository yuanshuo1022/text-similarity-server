import numpy as np
import jieba
from datasketch import MinHash, MinHashLSH


class SimilarityServer:
    # 计算句子向量
    def sentence_vector(sentence, word2vec_model, word2tfidf):

        words = jieba.lcut(sentence)
        sentence_vec = np.zeros(word2vec_model.vector_size)
        total_weight = 0  # 用于累计总权重
        for word in words:
            if word in word2vec_model.wv and word in word2tfidf:
                word_vec = word2vec_model.wv[word]
                tfidf_weight = word2tfidf[word]
                sentence_vec += word_vec * tfidf_weight
                total_weight += tfidf_weight
        if total_weight == 0:
            return sentence_vec / (total_weight + 0.001)  # 避免除以零
        return sentence_vec / total_weight

    # 定义将文本转化为平均向量的函数
    def text_to_average_vector(text, model):
        words = jieba.lcut(text)
        word_vectors = [model.wv[word] for word in words if word in model.wv]

        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # 定义 Jaccard 相似度函数
    def jaccard_similarity(text1, text2):
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        return intersection / union

    # 定义相关系数相似度函数
    def correlation_similarity(vec1, vec2):
        corr_coef = np.corrcoef(vec1, vec2)
        return corr_coef[0][1]

    # 定义局部敏感哈希（LSH）相似度函数
    def lsh_similarity(text1, text2, lsh):
        text1_tokens = set(jieba.lcut(text1))
        text2_tokens = set(jieba.lcut(text2))

        minhash1 = MinHash(num_perm=128)
        minhash2 = MinHash(num_perm=128)

        for word in text1_tokens:
            minhash1.update(word.encode('utf-8'))
        for word in text2_tokens:
            minhash2.update(word.encode('utf-8'))

        lsh.insert("text1", minhash1)
        lsh.insert("text2", minhash2)

        result = lsh.query(minhash1)

        return len(result) / len(text1_tokens)
