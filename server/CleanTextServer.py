class TextCleanServer:
    def clean_text_cn_en(texts):
        cleaned_text_cn_en = ''.join(char for char in texts if '\u4e00' <= char <= '\u9fff' or 'a' <= char.lower() <= 'z')
        return cleaned_text_cn_en

    def clean_text_cn(texts):
        cleaned_text_cn = ''.join(char for char in texts if '\u4e00' <= char <= '\u9fff')
        return cleaned_text_cn

    def clean_text_cn_en_num(texts):
        cleaned_text_cn_en = ''.join(char for char in texts if ('\u4e00' <= char <= '\u9fff') or (
                    'a' <= char.lower() <= 'z') or char.isdigit() or char == '.')
        return cleaned_text_cn_en
