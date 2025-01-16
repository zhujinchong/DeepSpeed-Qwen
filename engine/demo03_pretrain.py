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

from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, OPTForCausalLM
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling



def preprocess_dataset_to_batch(examples):
    global tokenizer
    # refer from https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_clm_pt_with_peft.py#L491
    tokenized_examples = tokenizer(examples['text'])
    # 将字符串拼接起来
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    # concatenated_examples:
    # input_ids [2, 50977, 53763,
    # attention_mask [1, 1, 1, 1, 1

    # input_ids 的长度
    total_length = len(concatenated_examples[list(tokenized_examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    block_size = 1024
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    return result


def preprocess_pretrain_dataset(corpus_path):
    file_list = []
    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file))

    train_dataset = load_dataset('text', data_files=file_list, split="train[10%:]")
    eval_dataset = load_dataset('text', data_files=file_list, split="train[:10%]")

    train_dataset.shuffle()
    train_dataset = train_dataset.map(preprocess_dataset_to_batch, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset.shuffle()
    eval_dataset = eval_dataset.map(preprocess_dataset_to_batch, batched=True, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset


def pretrain(corpus_path, model_path, save_path):
    global model
    global tokenizer
    model = OPTForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(output_dir=save_path, num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=2,
                                      optim="adamw_torch", lr_scheduler_type="cosine", learning_rate=1e-3, warmup_steps=0, logging_steps=1)
    train_dataset, eval_dataset = preprocess_pretrain_dataset(corpus_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # metrics=compute_metrics,
        # compute_loss_func=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    )
    print('*** Start training. ***')
    trainer_result = trainer.train()
    metrics = trainer_result.metrics
    print(metrics)
    # trainer.log_metrics('train', metrics)
    # trainer.save_metrics('train', metrics)
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    # 所有配置
    corpus_path = "../datasets/pretrain/example/train"
    model_path = "../checkpoints/facebook_opt_new_tokenizer"
    save_path = "../checkpoints/facebook_opt_pretrain"

    pretrain(corpus_path, model_path, save_path)
