import torch
from llamavid.eval.eval_dataset import ProductParquetEvalDataset, ProductDataEvalCollator
from llamavid.train.train_test import load_pretrained_model
import transformers

def eval_single(model, tokenizer, prompts, input_ids, products, answer):
    print("ground truth:", answer)
    with torch.inference_mode():
        model.update_prompt(prompts)
        outputs = model.generate(input_ids=input_ids, products=products)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated answer:", generated_text)  # 输出生成的答案


def evaluate_model(model, eval_dataset, tokenizer):
    training_args = transformers.TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
    )

    collator = ProductDataEvalCollator(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

# 测试代码
if __name__ == "__main__":
    model = load_pretrained_model(
        model_base="data/vicuna-7b-v1.5",
        model_path="data/output_model_with_adapter/"
    )
    data_path = "data/dataset/part-00000.parquet"
    
    tokenizer = transformers.BertTokenizer.from_pretrained("data/bert-base-uncased")
    eval_dataset = ProductParquetEvalDataset(data_path, tokenizer=tokenizer, prompt_type='single_product_title_prompt')
    
    print("Dataset stats:")
    print(eval_dataset.get_stats())
    
    print("\nSample data:")
    samples = eval_dataset.get_sample_data(3)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        eval_single(model, tokenizer, sample['prompts'], sample['input_ids'], sample['products'], sample['answer'])