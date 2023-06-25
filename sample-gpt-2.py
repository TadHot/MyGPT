from transformers import BertJapaneseTokenizer
path = './BERT-base_mecab-ipadic-bpe-32k'
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

text ="私が目指すのは、新しい資本主義の実現です。成長を目指すことは極めて重要であり、その実現に向けて全力で取り組みます。"

token = tokenizer.tokenize(text)
token

input_ids = tokenizer.encode(text)
input_ids
