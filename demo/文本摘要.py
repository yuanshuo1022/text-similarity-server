import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance

nltk.download('stopwords')
nltk.download('punkt')


def text_summarization(text, num_sentences=3):
    # 分句
    sentences = sent_tokenize(text)

    # 分词和去停用词
    stop_words = set(stopwords.words('chinese'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # 计算单词频率
    freq = FreqDist(words)

    # 创建句子向量
    sentence_vectors = []
    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)
        sent_vec = []
        for word in words:
            if word in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    # 计算句子相似度矩阵
    sim_matrix = []
    for i in range(len(sentences)):
        sim_scores = []
        for j in range(len(sentences)):
            if i != j:
                sim_scores.append(1 - cosine_distance(sentence_vectors[i], sentence_vectors[j]))
            else:
                sim_scores.append(0)
        sim_matrix.append(sim_scores)

    # 使用PageRank算法得出句子分数
    scores = pagerank(sim_matrix)

    # 获取前N个得分最高的句子作为摘要
    ranked_sentences = [(scores[i], sentences[i]) for i in range(len(sentences))]
    ranked_sentences.sort(key=lambda x: x[0], reverse=True)

    summary = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    return ' '.join(summary)


# PageRank算法
def pagerank(graph, damping_factor=0.85, max_iterations=100, min_diff=1e-5):
    num_nodes = len(graph)
    scores = [1.0 / num_nodes] * num_nodes
    for _ in range(max_iterations):
        new_scores = [0.0] * num_nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if graph[j][i] > 0:
                    new_scores[i] += scores[j] / sum(graph[j])
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(num_nodes))
        if diff < min_diff:
            return new_scores
        scores = [(1 - damping_factor) + damping_factor * s for s in new_scores]
    return scores


# 示例文本
sample_text = """
这是一段需要进行文本摘要的长篇文本。文本摘要的目标是提取关键信息，以减少文本长度。
"""

# 执行文本摘要
summary = text_summarization(sample_text)
print(summary)
