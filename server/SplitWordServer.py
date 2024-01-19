import jieba
import spacy


class SplitWord :
    #文本类型
    def jieba_split_word(text):
        # 使用结巴分词
        jieba_tokens = jieba.lcut(text)
        return  jieba_tokens

    # 初始化 spaCy 中文模型
    def spacy_split_word(text):
        nlp = spacy.load('zh_core_web_sm')
        doc = nlp(text)
        return [token.text for token in doc]

    #文件类型
    def jieba_split_word_file(self):
        return "等待开发"

    def spacy_split_word_file(self):
        return "等待开发"