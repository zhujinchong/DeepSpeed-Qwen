# -*- encoding: utf-8 -*-
"""
@File    :   demo01_expand_vocab_from_corpus.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2025/1/15 10:59    1.0         None
"""
import os
import shutil

import sentencepiece
from transformers import AutoTokenizer, AutoModel, OPTForCausalLM
import torch


def process_corpus(corpus_path):
    """加载 .txt 文件列表"""
    ret_list = []
    if not os.path.isdir(corpus_path):
        if not corpus_path.endswith('.txt') and not corpus_path.endswith('.tsv'):
            raise ValueError('Only .txt or .tsv files are supported.')
        else:
            ret_list.append(corpus_path)
    else:
        file_list = os.listdir(corpus_path)
        for file in file_list:
            if not file.endswith('.txt') and not corpus_path.endswith('.tsv'):
                raise ValueError('Only .txt or .tsv files are supported.')
            else:
                ret_list.append(os.path.join(corpus_path, file))
    return ret_list


def inject_vocab(model_path, save_path, corpus_list):
    print('Start injecting new vocabulary.')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_words = []
    for file in corpus_list:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        words = [line.strip() for line in lines]
        all_words.extend(words)

    tokenizer.add_tokens(all_words)
    tokenizer.save_pretrained(save_path)
    print(f'New vocabulary injected, new tokenizer is saved to {save_path}.')


def add_new_tokens_to_tokenizer(tokenizer, save_path):
    print('===Start adding new tokens.===')
    bpe_model = os.path.join(save_path, 'bpe_expand.model')
    sp_bpe = sentencepiece.SentencePieceProcessor()
    sp_bpe.load(bpe_model)

    raw_vocab = [sp_bpe.id_to_piece(id) for id in range(sp_bpe.get_piece_size())]
    # clean_vocab = list(set(filter(is_chinese, raw_vocab)))

    tokenizer.add_tokens(raw_vocab)
    tokenizer.save_pretrained(save_path)
    print(f'===New tokens added, new tokenizer is saved to {save_path}.===')


def eval_new_tokenizer(save_path):
    print("===分词测试===")
    new_tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
    print(new_tokenizer.tokenize("白日依山尽，黄河入海流"))
    print("===模型测试===")
    prompt = '你'
    inputs = new_tokenizer(prompt, return_tensors="pt")
    new_model = OPTForCausalLM.from_pretrained(save_path, trust_remote_code=True)
    print(new_model)
    generate_ids = new_model.generate(inputs.input_ids, max_length=30)
    decoded_text = new_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(decoded_text)


def resize_model_embedding_size(model_path, save_path):
    model = OPTForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    """修改embedding层，lm_head层，并均值初始化"""
    print('===Start resizing embedding.===')
    old_tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
    old_tokenizer_len = len(old_tokenizer)
    new_tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
    new_tokenizer_len = len(new_tokenizer)
    # # 新增的token和在原来token相对应的字典
    token_mapping = {}
    for i in range(old_tokenizer_len, new_tokenizer_len):
        # 使用 tokenizer 的 convert_ids_to_tokens 方法将索引转换为对应的 token
        token = new_tokenizer.convert_ids_to_tokens(i)
        # 原来的token
        input_ids = old_tokenizer(token, return_tensors="pt").input_ids[0]
        new_input_ids = input_ids[1:]
        token_mapping[i] = new_input_ids

    # 原始输入embedding
    old_embeddings = model.get_input_embeddings()
    old_hidden_size = old_embeddings.weight.size()[1]
    new_embedding = torch.nn.Embedding(new_tokenizer_len, old_hidden_size)

    # 将现有Embedding层的权重赋值给新的Embedding层
    new_embedding.weight.data[:old_tokenizer_len, :] = old_embeddings.weight.data[:old_tokenizer_len, :]

    # 剩余的Embedding层 采用均值初始化：均值是new_token的old_tokenizer分词的均值
    for new_token, original_tokens in token_mapping.items():
        original_embeddings = old_embeddings(original_tokens)
        mean_embedding = torch.mean(original_embeddings, dim=0)
        new_embedding.weight.data[new_token] = mean_embedding

    # 更换嵌入层
    model.set_input_embeddings(new_embedding)
    # print(model)

    # 新的lm_head
    lm_head = model.lm_head
    new_lm_head = torch.nn.Linear(in_features=old_hidden_size, out_features=new_tokenizer_len, bias=False)
    new_lm_head.weight.data[:old_tokenizer_len, :] = lm_head.weight.data[:old_tokenizer_len, :]

    # 新增
    for new_token, original_tokens in token_mapping.items():
        original = 0
        for i in original_tokens:
            original += lm_head.weight.data[i]
        mean_para = original / len(original_tokens)
        new_lm_head.weight.data[new_token] = mean_para

    # 替换模型原来的lm_head
    model.lm_head = new_lm_head
    print(model)

    # 模型保存
    model.save_pretrained(save_path)

    # print(f'===New model: {model}===')
    print(f'===Embedding resized, new model is saved to {save_path}.===')


if __name__ == '__main__':
    # 所有配置
    corpus_path = "../datasets/expand_vocab/"
    model_path = "../checkpoints/facebook_opt-125m"
    new_model_path = "../checkpoints/facebook_opt_new_tokenizer"

    # 加载预料
    # corpus_list = process_corpus(corpus_path)
    # 添加分词
    # inject_vocab(model_path, new_model_path, corpus_list)
    # 改变embedding大小
    # resize_model_embedding_size(model_path, new_model_path)
    """
    改变embedding后，直接使用模型会报错：- decoder.embed_tokens.weight: found shape torch.Size([59765, 768]) in the checkpoint and torch.Size([50272, 768]) in the model instantiated
    原因：model实例化embedding size为 torch.Size([50272, 768])，但是加载权重为torch.Size([59765, 768])
    解决：手动修改config.json中的 "vocab_size": 59765
    """
    # 测试
    eval_new_tokenizer(new_model_path)
    # print('Successfully!')
