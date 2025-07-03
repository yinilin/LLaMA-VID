# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import argparse
import yaml
import os
# replace_llama_attn_with_flash_attn()

from llamavid.train.train_test import train
from llamavid.train.train_test import ModelArguments, DataArguments, TrainingArguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='llamavid train')

    parser.add_argument('--model_config', default='config/model_config.yml', type=str)
    parser.add_argument('--train_config', default='config/train_config.yml', type=str)
    parser.add_argument('--data_config', default='config/data_config.yml', type=str)


    args = parser.parse_args()
    # Load configurations from YAML files
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    model_args = ModelArguments(**model_config)
    data_args = DataArguments(**data_config)
    training_args = TrainingArguments(**train_config)
    print("load argments done")
    print("model_args:", model_args)
    print("data_args:", data_args)
    print("training_args:", training_args)

    def list_directory_contents(directory_path):
        try:
            # 获取目录中的所有文件和文件夹
            entries = os.listdir(directory_path)

            print(f"Contents of '{directory_path}':")
            for entry in entries:
                print(entry)

        except FileNotFoundError:
            print(f"The directory '{directory_path}' does not exist.")
        except PermissionError:
            print(f"Permission denied to access the directory '{directory_path}'.")
    print("check directory contents")
    list_directory_contents("/mnt/yinilin/yinilin")

    train(model_args, data_args, training_args)
