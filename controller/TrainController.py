from flask import request, jsonify, Blueprint
from server.CleanTextServer import TextCleanServer
from server.SplitWordServer import SplitWord
from server.SimpleConvertServer import ConverterSimple

train_route = Blueprint('train_route', __name__)


@train_route.route('/api/clean-text', methods=['POST'])
def clean_text_route():
    try:
        data = request.get_json()
        input_text = data.get('text')
        mode = int(data.get('mode'))
        print(f"mode: {mode}\ntext: {input_text}")
        if not input_text:
            return jsonify({'code': 301, 'error': '无效输入'})
        if mode == 1:
            cleaned_text = TextCleanServer.clean_text_cn(input_text)
        elif mode == 2:
            cleaned_text = TextCleanServer.clean_text_cn_en(input_text)
        elif mode == 3:
            cleaned_text = TextCleanServer.clean_text_cn_en_num(input_text)
        else:
            return jsonify({'code': 302, 'error': "请切换模式"})
        return jsonify({'code': 200, 'cleaned_text': cleaned_text})
    except Exception as e:
        return jsonify({'code': 300, 'error': str(e)})


@train_route.route('/api/split-word', methods=['POST'])
def split_word_route():
    try:
        data = request.get_json()
        input_text = data.get('text')
        mode = int(data.get('split_word_mode'))

        if not input_text:
            return jsonify({'code': 301, 'error': '无效输入'})
        if mode == 1:
            split_text = SplitWord.jieba_split_word(input_text)
        elif mode == 2:
            split_text = SplitWord.spacy_split_word(input_text)
        else:
            return jsonify({'code': 302, 'error': "请切换模式"})
        return jsonify({'code': 200, 'split_text': split_text})
    except Exception as e:
        return jsonify({'error': str(e)})


@train_route.route('/api/simple-convert', methods=['POST'])
def simple_convert_route():
    try:
        data = request.get_json()
        input_text = data.get('text')

        if not input_text:
            return jsonify({'code': 301, 'error': '无效输入'})
        converter = ConverterSimple()
        simple_text = converter.convert_simple_chinese(input_text)

        return jsonify({'code': 200, 'simple_text': simple_text})
    except Exception as e:
        return jsonify({'code': 300, 'error': str(e)})
