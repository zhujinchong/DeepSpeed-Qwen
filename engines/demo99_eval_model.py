# -*- encoding: utf-8 -*-
"""
@File    :   demo99_eval_model.py.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2025/1/17 15:40    1.0         None
"""
import transformers

transformers.set_seed(1234)
model_name_or_path = "../checkpoints/ckpt/checkpoint-30"


def eval_new_tokenizer(save_path):
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
    prompt = '天龙八'
    inputs = new_tokenizer(prompt, return_tensors="pt")
    print(inputs['input_ids'])
    new_model = transformers.OPTForCausalLM.from_pretrained(save_path, trust_remote_code=True)
    generate_ids = new_model.generate(inputs.input_ids, max_length=30)
    decoded_text = new_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(decoded_text)


# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
# model = transformers.OPTForCausalLM.from_pretrained(model_name_or_path)

eval_new_tokenizer(model_name_or_path)
