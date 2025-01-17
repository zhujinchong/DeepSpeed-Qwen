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
from transformers import AutoTokenizer, AutoModel


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


def train_vocab(save_path, corpus, vocab_size=8000, max_sentence_length=24000, character_coverage=0.9995):
    """训练词表"""
    print('===Start training the vocabulary.===')
    sentencepiece.SentencePieceTrainer.train(
        # 只支持 txt 和 tsv 格式
        input=corpus,

        # 保存的模型前缀名
        model_prefix='bpe_expand',

        # 词表大小
        vocab_size=vocab_size,

        # 指定模型的字符覆盖率, 中文日文等推荐为 0.9995, 其余可以尝试 1.0
        character_coverage=character_coverage,

        # 分词算法
        model_type='bpe',

        # 是否将数字划分为单个 token, (LLama需要设置为True
        split_digits=False,

        # 指定在遇到未知或很少的字符时将其分解为 UTF-8 字节, 开启后等效于 bbpe
        byte_fallback=True,

        # 指定输入句子的最大长度，以字节为单位
        max_sentence_length=max_sentence_length,

    )
    shutil.move('bpe_expand.model', save_path)
    shutil.move('bpe_expand.vocab', save_path)
    print(f'===The vocabulary training is complete, saved to {save_path}.===')


def add_new_tokens_to_tokenizer(tokenizer, save_path):
    """tokenizer添加token"""
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


def resize_model_embedding_size(model, save_path):
    """model修改embedding size"""
    print('===Start resizing embedding.===')
    new_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer_length = len(new_tokenizer)

    new_length = int(tokenizer_length // 64 + 1) * 64
    model.resize_token_embeddings(new_length)

    model.save_pretrained(save_path)
    print(f'===New model: {model}===')
    print(f'===Embedding resized, new model is saved to {save_path}.===')


if __name__ == '__main__':
    # 所有配置
    corpus_path = "../datasets/pretrain/example/train/"
    model_path = "../checkpoints/facebook_opt-125m"
    new_model_path = "../checkpoints/facebook_opt_new_tokenizer"

    # 加载模型
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载预料
    corpus_list = process_corpus(corpus_path)
    # 分词
    train_vocab(new_model_path, corpus_list)
    # 添加分词
    add_new_tokens_to_tokenizer(tokenizer, new_model_path)
    # 测试
    eval_new_tokenizer(new_model_path)
    # 改变embedding大小
    resize_model_embedding_size(model, new_model_path)
    print('Successfully!')
