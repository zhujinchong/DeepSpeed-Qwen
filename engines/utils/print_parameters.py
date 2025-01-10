# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : copy from https://github.com/airaria/TextPruner/blob/main/src/textpruner/utils.py
# @Email : gzlishouxian@gmail.com
# @File : print_parameters.py
# @Software: PyCharm


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}')
