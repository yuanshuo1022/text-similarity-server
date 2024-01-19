import opencc


class ConverterSimple:
    def __init__(self):
        self.cc = opencc.OpenCC('t2s')

    def convert_simple_chinese(self, text):
        simple_text = self.cc.convert(text)
        return simple_text
