#    Copyright 2023 Haotian Liu
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
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LLaMAVIDMetaModel:

    def __init__(self, config):
        super(LLaMAVIDMetaModel, self).__init__(config)

        '''if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)'''
        self.config.mm_hidden_size = getattr(config, 'mm_hidden_size', 768)
        self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        ### 保留projection,去掉encoder
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter  

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.max_token = max_token
        
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_attention_modules(self, model_args, for_eval=False):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        self.config.bert_type = getattr(model_args, "bert_type", "raw_bert_layer:2")
        self.config.compress_type = getattr(model_args, "compress_type", None)

        att_feat_size = self.config.mm_hidden_size
        self.vlm_att_tokenlizer, self.vlm_att_encoder = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.vlm_att_encoder.config.hidden_size, self.config.mm_hidden_size)  ### 从text decoder 到 context attention的映射，得到text query
        self.vlm_att_key_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)  ### 从vision encoder到 context attention中的projection，是跨注意力的一部分
        self.vlm_att_val_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)  ### 计算完attention后的projection部分

        if "raw" in self.config.bert_type:
            ### 代表Xt到bert到projection，必要
            self.vlm_att_bert_proj  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder.config.hidden_size)  ### 用于将vision embedding到text decoder维度的映射
        else:
            self.vlm_att_bert_proj = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'freeze_all' in self.config.bert_type:
            print("Freeze bert and attention weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_projector.requires_grad_(False)
            self.vlm_att_key_projector.requires_grad_(False)
            self.vlm_att_val_projector.requires_grad_(False)
        elif 'freeze' in self.config.bert_type:
            print("only freeze bert weights...")
            self.vlm_att_encoder.requires_grad_(False)

        if pretrain_mm_mlp_adapter is not None:   ### att_projector_weights包含vlm_att_projector、vlm_att_key_projector、vlm_att_val_projector
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            trainable_module = ['vlm_att_encoder', 'vlm_att_projector', 'vlm_att_key_projector', 
                                'vlm_att_val_projector']
            if hasattr(model_args, 'model_name_or_path'):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path
            model_idx_path = getattr(model_args, 'model_path', model_save_path)
            ###  增加处理，对没有pytorch_model.bin.index.json的情况
            try:
                weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
                model_path = set([weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
                att_projector_weights = {}
                for _model in model_path:
                    att_projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
                if len(att_projector_weights) == 0:
                    return
            except FileNotFoundError:
                print(f"Warning: No pytorch_model.bin.index.json found in {model_idx_path}, using randomly initialized attention modules.")
                return
        
        bert_dict = get_w(att_projector_weights, 'vlm_att_encoder')
        if "bert.embeddings.position_ids" not in bert_dict and "raw_bert" not in self.config.bert_type:
            bert_dict["bert.embeddings.position_ids"] = self.vlm_att_encoder.bert.embeddings.position_ids
        print('Loading pretrained weights...')
        self.vlm_att_encoder.load_state_dict(bert_dict)
        self.vlm_att_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_projector'))
        self.vlm_att_key_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_key_projector'))
        self.vlm_att_val_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_val_projector'))
        
        if for_eval:
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_key_projector = self.vlm_att_key_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_val_projector = self.vlm_att_val_projector.to(device=device_type, dtype=weight_type)
            

    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("data/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained("data/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        
        if "raw" in self.config.bert_type:
            encoder_config.is_decoder = True
            mm_model = BertLMHeadModelRaw.from_pretrained(
                "data/bert-base-uncased", config=encoder_config
            )
        else:
            raise NotImplementedError("BERT type not implemented...")
        
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        if "layer" in self.config.bert_type:
            layer_num = int(self.config.bert_type.split(':')[-1])
            mm_model.bert.encoder.layer = mm_model.bert.encoder.layer[:layer_num]
            print(f"Only use {layer_num} layers in BERT...")
        
        return tokenizer, mm_model


class LLaMAVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, product_features, prompts=None, image_counts=None):  
        ### 为每个图像生成n+1个特征token   
        ### product_features: (batch*num_product, embedding_dim)
        ### 每个products对应两个token，返回形状[prompt数, frame数 * n_token, C]
        ### 这里的prompt是文本，还没有tokenize
        product_features = self.vlm_attention(product_features, 
                                            prompts=prompts, 
                                            image_counts=image_counts)
        return product_features

    def vlm_attention(self, product_features, prompts=None, image_counts=None):     
        ### product_features: (batch*num_product, product_seq_len, embedding_dim)
        ### prompts: List[List[str]]，每个batch对应一个prompt，可能包含多轮对话
        ### 要注意每个一个batch里的图片数量可能不一样，所以需要image_counts来指明
        ### image_counts： [num_frames1, num_frames2, ...]，每个batch的图像数量
        ### 该函数的目的是提取每一个图像的context token和content token，形成列表返回
        img_feat_lst = []
        if image_counts is None:
            assert len(product_features) == len(prompts), f"Size mismatch! product_features: {len(product_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(product_features.size()[:-1], dtype=torch.long).to(product_features.device)   ### attention矩阵，

        total_count = 0
        # calculate each image feat according to the prompt
        for _idx in range(len(prompts)):
            assert (len(prompts[_idx]), 1), f"Prompt should be a list with length 1, but got {type(prompts[_idx])} with length {len(prompts[_idx])}."
            ### prompts[_idx]也是一个列表
            input_token = self.get_model().vlm_att_tokenlizer(  ### 这里是bert的tokenizer
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(product_features.device)

            input_ids = input_token.input_ids  ### (prompt_sentence_num, seq_len)
            attention_masks = input_token.attention_mask
            
            if image_counts is None:
                img_feat_prompt = product_features[_idx, None].expand(len(prompts[_idx]), -1, -1)  ### (prompt_sentence_num, product_seq_len, feat_dim)
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)  ### (prompt_sentence_num, product_seq_len)
            else:
                # shape: [prompt_num*frame_num, product_seq_len, feat_dim]
                img_feat_prompt = product_features[total_count:total_count+image_counts[_idx]]  ### 得到image_counts[_idx]个图像的特征
                img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0,1)  ### (prompt_sentence_num*num_product, product_seq_len, feat_dim)
                img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0,1)
                input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)  ### (prompt_sentence_num*num_product, seq_len),把prompt里每个语句的tokens重复num_product次
                attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                total_count += image_counts[_idx]
            
            bert_feat = img_feat_prompt.clone()
            
            mm_output = self.get_model().vlm_att_encoder.bert(
                input_ids,
                attention_mask=attention_masks,
                encoder_hidden_states=self.get_model().vlm_att_bert_proj(bert_feat),
                encoder_attention_mask=img_att_prompt,
                return_dict=True,
            )
            
            mm_output = mm_output.last_hidden_state
            text_q = self.get_model().vlm_att_projector(mm_output)
            final_token = self.token_generation(text_q, img_feat_prompt)

            if image_counts is not None:
                # shape: [product_num，n, feat_dim]
                final_token = final_token.reshape(len(prompts[_idx]), image_counts[_idx], *final_token.shape[-2:])
                final_token = final_token.squeeze(0)
            img_feat_lst.append(final_token)

        return img_feat_lst

    def token_generation(self, text_q, vis_embed, long_video=False):
        ctx_embed = self.get_model().vlm_att_key_projector(vis_embed)
        # Key part 1: calculate context-related embedding
        ctx_embed = text_q @ ctx_embed.transpose(-1,-2) 
        ctx_embed = ctx_embed / (vis_embed.shape[-1] ** 0.5)
        if not long_video:
            ctx_embed = (ctx_embed.softmax(-1) @ vis_embed).mean(1)
        else:
            block_size = 64
            outputs = []
            ctx_score = ctx_embed.softmax(-1)    
            for L in range(0, len(ctx_score), block_size):
                R = L + block_size
                sub_embed = (ctx_score[L:R] @ vis_embed[L:R]).mean(1)
                outputs.append(sub_embed)
            ctx_embed = torch.cat(outputs)
            torch.cuda.empty_cache()
        ctx_embed = self.get_model().vlm_att_val_projector(ctx_embed[:,None])

        # Key part 2: calculate visual embedding
        if self.config.compress_type is not None:
            if 'grid' in self.config.compress_type:
                grid_size = int(self.config.compress_type.split('grid:')[-1])
                cur_shape = int(vis_embed.shape[1]**0.5)
                assert grid_size > 1, f'Grid size should be larger than 1, but got {grid_size}'
                vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
                grid_stride = cur_shape // grid_size
                vis_embed = F.avg_pool2d(vis_embed.permute(0, 3, 1, 2), 
                                         padding=0,
                                         kernel_size=grid_stride, 
                                         stride=grid_stride)
                
                vis_embed = vis_embed.permute(0, 2, 3, 1).flatten(1,2)
            elif 'mean' in self.config.compress_type:
                vis_embed = vis_embed.mean(dim=1, keepdim=True)
        
        # concat token in shape (B, n+1, C)
        vis_embed = self.get_model().mm_projector(vis_embed)                
        final_token = torch.cat([ctx_embed, vis_embed], dim=1)
        return final_token

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, products, prompts=None
    ): 
        
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts
        ### 这里的prompt是原文本，不是input_ids，是用来在text decoder做融合的，input_ids里有image位置的标记，所以不能直接用。
        ### 这里的输入特征已经是encoder后的向量
        ### products形状:(num_products, product_seq_len, embedding_dim)的列表，列表长度为batch，image_counts用来处理每个batch中商品数量不同
        if type(products) is list:
            image_counts = [i.shape[0] for i in products]
            concat_images = torch.cat(products, dim=0)  ### 形状为(batch*num_product, product_seq_len, embedding_dim)
            image_features = self.encode_images(concat_images, prompts, image_counts)
        else:
            ### 如果products不是列表，只有一个batch，就不需要concat
            image_features = self.encode_images(products, prompts)

        ### image_features:[(num_product，n(受compresss_type影响), 4096),……]
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            ### 进行图像特征和文本特征拼接
            token_idx = 0  ### image占位符计数
            while image_token_indices.numel() > 0:
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][token_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                token_idx += 1
            
            # changle image idx after processing one sample
            cur_image_idx += 1
            if cur_input_ids.numel() > 0:  ### 处理完所有image token后，剩余的文本token
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        ### input_embeds: (batch, seq_len, hidden_size)
        return None, attention_mask, past_key_values, new_input_embeds, new_labels
