from datasets import load_dataset

def prepare_and_save_data(tokenizer, output_csv_path="sample_data/train.csv", num_samples=3000):
    """
    Load, format prompt and save training dataset to CSV file.
    """
    print("Downloading BANKING77...")
    dataset = load_dataset("banking77")
    
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_samples))
    label_names = train_ds.features["label"].names

    prompt = """### Instruction:
Classify the intent of the following banking query. Return only the intent label.

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["text"]
        labels = examples["label"]

        texts = []
        for input_text, label_id in zip(inputs, labels):
            label_name = label_names[label_id]
            text = prompt.format(
                input_text.strip(),
                label_name.strip()
            ) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    print("Formatting prompt...")
    train_ds = train_ds.map(formatting_prompts_func, batched=True)

    train_ds.to_csv(output_csv_path, index=False)
    print(f"Done! Saved {len(train_ds)} samples to file: {output_csv_path}")

    return train_ds