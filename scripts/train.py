import torch
import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

max_seq_length = 2048

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_prepare_data():
    dataset = load_dataset("csv", data_files="sample_data/train.csv")
    train_ds = dataset["train"]
    
    return train_ds

def load_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['model_name'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,
        load_in_4bit=config['model']['load_in_4bit'],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer

def main():
    config = load_config("configs/train.yaml")
    
    print("Loading data from sample_data/train.csv...")
    train_ds = load_and_prepare_data()

    print("Loading model...")
    model, tokenizer = load_model(config)

    print("Training...")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            warmup_ratio = config['training']['warmup_ratio'],
            num_train_epochs = config['training']['num_train_epochs'], 
            learning_rate = config['training']['learning_rate'],
            logging_steps = config['training']['logging_steps'], 
            optim = config['training']['optim'],
            weight_decay = config['training']['weight_decay'],
            lr_scheduler_type = config['training']['lr_scheduler_type'],
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    trainer.train()

    print("Saving model...")
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])

    print(f"Done! Model saved to {config['training']['output_dir']}/")

if __name__ == "__main__":
    main()