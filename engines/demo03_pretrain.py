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
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import transformers

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class PretrainConfig:
    model_name_or_path: Optional[str] = field(default='../checkpoints/facebook_opt_new_tokenizer', metadata={"help": "Path to pretrained model checkpoint"})
    train_file_path: Optional[str] = field(default='../datasets/pretrain/example/train/tianlongbabu_small.json', metadata={"help": "Path to train data file/directory"})
    max_length: int = field(default=1024, metadata={"help": "Max length of input"})
    text_key_name: Optional[str] = field(default="content", metadata={"help": "key to text field name in train and validation file"})
    preprocess_num_workers: int = field(default=1, metadata={"help": "The number of processes to use for the preprocessing."})


@dataclass
class MyTrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default='../checkpoints/ckpt')
    do_train: bool = True
    do_eval: bool = True
    overwrite_output_dir: bool = True
    num_train_epochs: int = 3
    learning_rate: float = 3e-5
    evaluation_strategy: str = 'steps'
    eval_steps: int = 10
    save_strategy: str = 'steps'
    save_steps: int = 10


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise ValueError(f"Path: {path} not exists!")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)


def main():
    transformers.set_seed(1234)

    parser = transformers.HfArgumentParser((PretrainConfig, MyTrainingArguments))
    pretrain_config, training_args = parser.parse_args_into_dataclasses()

    # check file existence
    if pretrain_config.train_file_path:
        check_file_exist(pretrain_config.train_file_path)

    # load model, tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrain_config.model_name_or_path, padding_side='right', trunction_side="right", max_length=pretrain_config.max_length)
    model = transformers.OPTForCausalLM.from_pretrained(pretrain_config.model_name_or_path)

    # Split 20% of train data as validation data
    train_ds, validation_ds = datasets.load_dataset('json', data_files=pretrain_config.train_file_path, split=['train[:80%]', 'train[80%:]'])
    raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})

    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets["validation"].column_names

    def process_pretrain(batch):
        texts = batch[pretrain_config.text_key_name]
        texts = [text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token for text in texts]
        tokenized = tokenizer(texts)
        for k in tokenized.keys():
            batch[k] = [e[:pretrain_config.max_length] for e in tokenized[k]]
        batch['labels'] = batch['input_ids'].copy()
        return batch

    with training_args.main_process_first(desc="Process pretrain dataset"):
        clm_dataset = raw_datasets.map(
            process_pretrain,
            batched=True,
            batch_size=2,
            num_proc=pretrain_config.preprocess_num_workers,
            remove_columns=column_names,
            desc="Process pretrain dataset"
        )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=clm_dataset["train"],
        eval_dataset=clm_dataset["validation"],
        tokenizer=tokenizer,  # trainer need tokenizer.pad_token_id,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=pretrain_config.max_length,
                                                                      label_pad_token_id=-100),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # trigger Training
    trainer.train()
    # trainer.save_model()
    # trainer.save_state()


if __name__ == '__main__':
    main()

"""
deepspeed \
--include="localhost:0,1,2,3" \
./demo03_pretrain.py \
--deepspeed ../deepspeed_configs/ds_config_zero3.json \
--model_name_or_path ../checkpoints/facebook_opt_new_tokenizer \
--train_file_path ../datasets/pretrain/example/train/tianlongbabu.json \
--do_train \
--output_dir ./ckpt-clm \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 5 \
--learning_rate 1e-5 \
--evaluation_strategy steps \
--eval_steps 10 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2
"""
