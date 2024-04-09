
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import jsonlines

model_path = ''
model = torch.load(model_path, map_location=torch.device(''))
tokenizer = BertTokenizer.from_pretrained('', do_lower_case=False)

def tensor_to_numpy(emb):
    # 将向量补充到统一长度
    target_second_dim = 512
    current_second_dim = emb.size(1)
    padding_size = target_second_dim - current_second_dim
    emb = F.pad(emb, (0, 0, 0, padding_size)) if padding_size > 0 else emb
    ## 改变向量维度
    emb = emb.reshape(-1)   # 128 * 6(vocab_size)实体类型数量
    ## tensor转numpy
    emb = emb.detach().numpy().tolist()   ### 最终转为数组
    return emb

def cosine_similarity(v1, v2):
    def dot_product(v1, v2):
        return sum(x * y for x, y in zip(v1, v2))

    def magnitude(vector):
        return sum(x ** 2 for x in vector) ** 0.5
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2))

def get_database(path, database_path):
    database = []
    file = jsonlines.open(path)
    # with open(path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        dict = line
        sentence = dict['text']  # 去除首尾的空白字符
        tokens = tokenizer.tokenize(sentence)
        xx = tokenizer.convert_tokens_to_ids(tokens[:512])
        x = torch.LongTensor([xx])
        emb, _, _ = model(x, x)
        emb = tensor_to_numpy(emb)
        database.append([emb, dict])   ##数据库每一项[emb, dict]

    with jsonlines.open(database_path, 'w') as writer:
        writer.write_all(database)
    return database

def choose_demonstrations(sentence, database_path):
    database = jsonlines.open(database_path)
    tokens = tokenizer.tokenize(sentence)
    xx = tokenizer.convert_tokens_to_ids(tokens[:512])
    x = torch.LongTensor([xx])
    emb, _, _ = model(x, x)
    emb = tensor_to_numpy(emb)
    demonstrations_dict = {0: '', -1: '', -2: ''}
    for data in database:   # [emb, data]
        out = data[1]
        similarity = cosine_similarity(data[0], emb)
        demonstrations_dict[similarity] = out
        sorted_numbers = sorted(demonstrations_dict.keys(), reverse=True)[:3]
        demonstrations_dict = {num: demonstrations_dict[num] for num in sorted_numbers}
    return demonstrations_dict
