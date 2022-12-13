import re
import string

chinese_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
english_punctuation = string.punctuation


def pre_question(question, max_ques_words):
    question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

    question = re.sub(
        r"\s{2,}",
        ' ',
        question,
    )
    question = question.rstrip('\n')
    question = question.strip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = caption.lower().lstrip(chinese_punctuation).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def pre_chinese(text):

    text = text.lower().replace(chinese_punctuation, " ").replace(english_punctuation, " ")

    text = re.sub(
        r"\s{2,}",
        ' ',
        text,
    )
    text = text.rstrip('\n')
    text = text.strip(' ')

    return text