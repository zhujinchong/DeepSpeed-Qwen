# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
from transformers import TextIteratorStreamer
from engines.utils.metrics import Metrics
from engines.models import BaseModels
from engines.utils.logits_process import logits_processor
from threading import Thread


class Predictor(BaseModels):
    def __init__(self, data_manager, config, logger):
        super().__init__(data_manager, config, logger)
        self.logger = logger
        self.data_args = config.data_args
        self.generating_args = config.generating_args
        self.data_manager = data_manager
        self.prompt_template = data_manager.prompt_template
        self.metrics = Metrics(data_manager, logger)
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        self.model = self.load_base_model()
        self.model = self.load_adapter(self.model, adapter_dir=self.model_args.checkpoint_dir)
        self.logger.info(f'Model struct:\n{self.model}')
        self.model.eval()

    def terminal_inference(self):
        def predict(input, history):
            prompt_template = self.prompt_template.get_prompt(input, history)
            input_ids = self.tokenizer([prompt_template], return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs.update({
                'input_ids': input_ids,
                'eos_token_id': [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
                'logits_processor': logits_processor(),
                'streamer': streamer
            })
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            print(f'{self.model_args.model_type}:', end='', flush=True)
            response = ''
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response += new_text
            history = history + [(query, response)]
            return history

        history = []
        print('use `clear` to remove the history, use `exit` to exit the application.')
        while True:
            try:
                query = input('\nUser: ')
            except UnicodeDecodeError:
                print('Detected decoding error at the inputs, please set the terminal encoding to utf-8.')
                continue
            except Exception:
                raise
            if query.strip() == 'exit':
                break
            if query.strip() == 'clear':
                history = []
                print('History has been removed.')
                continue
            history = predict(query, history)
