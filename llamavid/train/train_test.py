# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# ------------------------------------------------------------------------
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import pickle
import math

import torch

import transformers

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llamavid.train.dataset import make_supervised_data_module
from llamavid.train.llava_trainer import LLaVATrainer

from llamavid import conversation as conversation_lib
from llamavid.model import *

from PIL import Image
from decord import VideoReader, cpu


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    bert_type: Optional[str] = field(default="qformer_pretrain")
    compress_type: Optional[str] = field(default=None)
    bert_path: Optional[str] = field(default="/mnt/yinilin/yinilin/models/bert-base-uncased", metadata={"help": "Path to the BERT model."})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    input_prompt: Optional[str] = field(default=None)
    max_seq_length: int = field(default=512)
    mm_hidden_size: int = field(default=48)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def get_mm_adapter_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        # keys_to_match = ['mm_projector']
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_att']  ### tofix：vision_resampler是什么？

        weight_to_save = get_mm_adapter_state(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train(model_args, data_args, training_args):
    global local_rank

    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.mm_hidden_size = data_args.mm_hidden_size

    '''model = LlavaLlamaAttForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        local_files_only=True
    )'''
    model = LlavaLlamaAttForCausalLM(config=config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp,
        max_token=training_args.model_max_length
    )
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # all the attention modules require grad
    model.get_model().initialize_attention_modules(model_args)

    # dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module
                    )

    # trainer.train()
    
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir="data/output_model_with_adapter")

def load_pretrained_model(model_base: str, model_path: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base)
    cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path)
    model = LlavaLlamaAttForCausalLM.from_pretrained(model_base, config=cfg_pretrained, local_files_only=True)

    mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    model.load_state_dict(mm_projector_weights, strict=False)

    return tokenizer, model

if __name__ == "__main__":
    # train()

    model_args = ModelArguments(
        model_name_or_path="data/vicuna-7b-v1.5",
        bert_type="raw_bert_layer:2",
        compress_type = "mean",
        tune_mm_mlp_adapter = True,
        bert_path="data/bert-base-uncased"
    )


    training_args = TrainingArguments(
        output_dir="data/output_model_with_adapter",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        # use_mps_device= True,
        save_steps=100,
        save_total_limit=1
    )
    
    data_args = DataArguments(
        data_path="data/dataset/part-00000.parquet",
        input_prompt="single_product_title_prompt",
        max_seq_length=512,
        mm_hidden_size=48
    )
    train(model_args, data_args, training_args)
