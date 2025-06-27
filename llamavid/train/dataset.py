from typing import Dict, Any, List
import torch
import transformers
import json
from torch.utils.data import Dataset
import pandas as pd
from pandas import read_parquet
from prompts import *
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dataclasses import dataclass, field

prompt_map = {
    'single_product_title_prompt': single_product_title_prompt
}

def convert_base64(b64_string: str) -> List[int]:
    """
    Convert a base64 string to a list of integers.
    """
    import base64
    import numpy as np

    # Step 1: Decode Base64 to bytes
    byte_data = base64.b64decode(b64_string)

    # Step 2: Convert bytes to float16 array
    float_array = np.frombuffer(byte_data, dtype=np.float16)

    return torch.tensor(float_array)

class ProductParquetDataset(Dataset):
    """Dataset for reading parquet files with product data and converting to conversation format."""
    ### TOFIX: 只支持单产品嵌入
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer = None,
                 prompt_type: str = 'single_product_title_prompt',
                 data_args = None):
        super(ProductParquetDataset, self).__init__()
        
        # 读取 parquet 文件
        self.df = pd.read_parquet(data_path)
        
        # 检查必要的字段是否存在
        required_fields = ['ext_titl', 'item_site_id', 'aspct_vlu_nm', 'unique_id', 'price']
        missing_fields = [field for field in required_fields if field not in self.df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields in parquet file: {missing_fields}")
        
        # 只保留需要的字段
        self.df = self.df[required_fields].copy()
        
        # 处理缺失值
        self.df = self.df.fillna('')  # 将 NaN 填充为空字符串
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        print(f"Loaded {len(self.df)} records from {data_path}")

        # 生成prompt message
        self.prompt_template = prompt_map.get(prompt_type, single_product_title_prompt)

    def _build_data(self, row) -> Dict:
        """将产品数据转换为输入格式"""
        # 构建产品信息字符串
        product_info = {
            'title': str(row['ext_titl']),
            'site_id': str(row['item_site_id']),
            'embedding': str(row['aspct_vlu_nm']),
            'unique_id': str(row['unique_id']),
            'price': str(row['price'])
        }
        user_prompts = self.prompt_template["question"].format(**product_info)
        assistance_answer = self.prompt_template["answer"].format(**product_info)
        prompt_chunks = user_prompts.split('<image>')
        user_messages = [self.tokenizer(prompt, add_special_tokens=False).input_ids for prompt in prompt_chunks]
        assistant_message =  self.tokenizer(assistance_answer, add_special_tokens=False).input_ids
        # 合并user_messages
        # 修改 insert_separator 函数，确保输出扁平化的列表
        def insert_separator(X, sep):
            if not X:  # 如果列表为空
                return []
            
            result = []
            for i, sublist in enumerate(X):
                result.extend(sublist)  # 添加当前子列表的所有元素
                if i < len(X) - 1:  # 如果不是最后一个元素
                    result.append(sep)  # 添加分隔符
            return result
        user_messages = insert_separator(user_messages, IMAGE_TOKEN_INDEX)
        input_ids = user_messages + [self.tokenizer.pad_token_id] + assistant_message
        labels = [self.tokenizer.pad_token_id] * len(user_messages) + [self.tokenizer.pad_token_id] + assistant_message
        
        # padding
        max_length = self.data_args.max_seq_length if self.data_args else 512
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        # prompt
        modify_prompt = user_prompts.replace('<image>', '').strip()
        # modify_prompt += assistance_answer.strip() 不应该加上answer的内容。

        # product，（n,product_seq_len,emb_dim）
        # tofix:以后n可能不止1
        products = convert_base64(product_info['embedding']).unsqueeze(0)

        data_dict = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompts": [modify_prompt],
            "products": products
        }

        return data_dict


    def __len__(self):
        return len(self.df)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        返回第 i 条数据，格式与LazySupervisedDataset一致
        """
        data = self.df.iloc[i]          
        instructions = self._build_data(dict(data))
        return instructions

    def get_sample_data(self, n=5):
        """获取前 n 条数据的示例"""
        samples = []
        for i in range(min(n, len(self))):
            samples.append(self.df.iloc[i])
        return samples

    def get_stats(self):
        """获取数据集的统计信息"""
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return stats

@dataclass
class ProductDataCollator(object):
    """数据收集器，用于批处理"""

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """将多个样本组合成一个batch"""
        # 收集input_ids和labels
        input_ids = [instance['input_ids'] for instance in instances if 'input_ids' in instance]        
        labels = [instance['labels'] for instance in instances if 'labels' in instance]
        
        if len(input_ids) == 0:
            return {}
        
        # 填充到相同长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)
        }

        # 如果有产品数据
        if 'products' in instances[0]:
            batch["products"] = [instance['products'] for instance in instances]
        else:
            batch["products"] = [torch.tensor([], dtype=torch.float16) for _ in instances]

        # 如果有prompt数据
        if 'prompts' in instances[0]:
            batch['prompts'] = [instance['prompts'] for instance in instances]
        else:
            batch['prompts'] = [[''] for _ in instances]
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """创建监督学习数据模块"""
    train_dataset = ProductParquetDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        prompt_type=data_args.input_prompt
    )
    
    data_collator = ProductDataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


# 测试代码
if __name__ == "__main__":
    # 使用示例
    data_path = "data/dataset/part-00000.parquet"  # 替换为你的parquet文件路径
    
    # 创建数据集
    tokenizer = transformers.BertTokenizer.from_pretrained("data/bert-base-uncased")
    dataset = ProductParquetDataset(data_path, tokenizer=tokenizer, prompt_type='single_product_title_prompt')
    
    # 查看数据集信息
    print("Dataset stats:")
    print(dataset.get_stats())
    
    # 查看几条示例数据
    print("\nSample data:")
    samples = dataset.get_sample_data(3)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        print()
    sample = dataset.__getitem__(1)
    print(sample)

    # 测试data_collator
    collator = ProductDataCollator(tokenizer=tokenizer)
    batch = collator([dataset[i] for i in range(3)])  # 获取前3条数据的batch
    print("\nBatch data:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")