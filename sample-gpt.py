from transformers import BertTokenizer

# トークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# テキストのトークン化と数値化
text = "私　は　猫　と　犬　が　好き　です。"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokens)     # ['i', 'love', 'cats', 'and', 'dogs', '.']
print(token_ids)  # [1045, 2293, 8801, 1998, 5959, 1012]
