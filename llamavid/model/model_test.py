from llamavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM
from transformers import LlamaConfig
import transformers
import torch
from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class LlavaConfig(LlamaConfig):
    model_type = "llava"

if __name__ == "__main__":
    from llamavid.train.train import ModelArguments
    model_args = ModelArguments(
        model_name_or_path="data/vicuna-7b-v1.5",
        bert_type="raw_bert_layer:2",
        compress_type = "mean"
    )
    print(model_args)
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    # print(config)
    model = LlavaLlamaAttForCausalLM(config)  # 创建模型
    # print(model)  # 打印模型

    # 构造随机输入
    batch = 2
    image_counts = [2, 3]
    product_seq_len = 3
    total_image = sum(image_counts)
    product_features = torch.randn(total_image, product_seq_len, 768)
    prompts = [["What is the product?"], ["Describe the features of the product."]]
    model.get_model().initialize_attention_modules(model_args)
    '''test1 = model.vlm_attention(product_features, prompts, image_counts)
    # 输出[(num_product, n, 4096),……]，列表长度为batch
    # compression为none时,n=product_seq_len+1;mean时，n=2；grid:grid+1
    print("test1", len(test1), test1[0].shape, test1[1].shape)  # 打印输出的形状
    ### 输出成功！'''



    # 假设 batch_size = 2，seq_len = 6，产品数分别为2和1，embedding_dim = 768
    batch_size = 2
    seq_len = 6
    product_seq_len = 3
    embedding_dim = 768

    # input_ids: batch 内每个样本的 token id
    input_ids = [
        torch.tensor([101, 102, IMAGE_TOKEN_INDEX, 103, 104, 105]),  # 第1个样本有2个image token
        torch.tensor([IMAGE_TOKEN_INDEX, 202, 203, 204, 205, 206])                 # 第2个样本有1个image token
    ]
    input_ids = torch.stack(input_ids, dim=0)

    # attention_mask: batch 内每个样本的 attention mask
    attention_mask = [
        torch.ones(seq_len, dtype=torch.long),
        torch.ones(seq_len, dtype=torch.long)
    ]
    attention_mask = torch.stack(attention_mask, dim=0)

    # labels: 与 input_ids 对齐
    labels = [
        torch.tensor([1, 1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1, 1, 1, 1]),  # -100 表示 image token 处不计算loss
    ]
    labels = torch.stack(labels, dim=0)

    # products: batch 内每个样本的产品特征，不需要堆叠会自动计算image_counts
    products = [
        torch.randn(2, product_seq_len, embedding_dim),  # 第1个样本有2个产品
        torch.randn(1, product_seq_len, embedding_dim)   # 第2个样本有1个产品
    ]


    # prompts: batch 内每个样本的 prompt，每个batch写一个prompt，多了不知道会发生什么
    prompts = [
        ["Describe the product."],
        ["What is this?"]
    ]

    # past_key_values: 一般推理时用，这里测试可设为None
    past_key_values = None

    # 调用
    # output = model.forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, labels=labels, products=products, prompts=prompts, return_dict = False)
    # print("Outputs:", output)
    
    with torch.inference_mode():
        model.update_prompt(prompts)
        answer = model(input_ids=input_ids, products=products)
        print("Generated answer:", answer.shape)  # 输出生成的答案形状

    # input_ids, attention_mask, past_key_values, inputs_embeds, labels = model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, products, prompts)
    # print("input_embeds shape:", inputs_embeds.shape)  # 输出输入嵌入的形状
    # print("labels", labels)  # 输出标签的形状
    # print("attention_mask shape:", attention_mask.shape)  # 输出注意力掩码的形状

    
