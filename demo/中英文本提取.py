from gensim.corpora import WikiCorpus
import jieba
from opencc import OpenCC  # 引入OpenCC


def convert_to_simplified(text):
    cc = OpenCC('t2s')  # 创建一个转换器，从繁体转为简体
    return cc.convert(text)


def tokenize(text, token_min_len=None, token_max_len=None, lower=None):
    text = convert_to_simplified(text)  # 调用转换函数将繁体中文转为简体中文
    seg_list = list(jieba.cut(text))  # 分词
    return seg_list


def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduce_zhiwiki.txt', 'w', encoding='utf-8')

    wiki = WikiCorpus(zhwiki_name, tokenizer_func=tokenize, dictionary={})  # 从xml文件中读出训练语料
    for text in wiki.get_texts():
        l.extend(text)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if (i % 200 == 0):
            print('Saved ' + str(i) + ' articles')
    f.close()


if __name__ == '__main__':
    my_function()
