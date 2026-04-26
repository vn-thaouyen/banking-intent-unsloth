import yaml
import torch
import pandas as pd
from tqdm import tqdm
import time
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config["checkpoint_path"],
            max_seq_length = self.config['max_seq_length'],
            dtype = None,
            load_in_4bit = self.config['load_in_4bit']
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        prompt = f"""### Instruction:
Classify the intent of the following banking query. Return only the intent label.

### Input:
{message}

### Response:
"""
        inputs = self.tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens = 128, use_cache = True)
        decoded_output = self.tokenizer.batch_decode(outputs)[0]

        predicted_label = decoded_output.split("### Response:")[-1].strip()
        predicted_label = predicted_label.split("\n")[0]
        return predicted_label

def clean(text):
    return (
        text.lower()
            .replace("<|end_of_text|>", "")
            .replace("</s>", "")
            .strip()
            .split("\n")[0]
    )

if __name__ == "__main__":
    model_path = "configs/inference.yaml"
    classifier = IntentClassification(model_path)
    
    print("="*50)
    print("   PART 1: 1 EXAMPLE INPUT   ")
    print("="*50)
    
    sample_query = "What happens if my card is stuck in the ATM?"
    print(f"User Query : {sample_query}")
    print("Predicting...")
    
    raw_result = classifier(sample_query)
    final_result = clean(raw_result)
    
    print(f"--> Intent : {final_result}")
    print("="*50 + "\n")
    
    time.sleep(2) 
    
    print("="*50)
    print("   PART 2: EVALUATION ON TEST.CSV   ")
    print("="*50)
    
    test_file_path = "sample_data/test.csv"
    try:
        df = pd.read_csv(test_file_path)
        total_samples = len(df)
        correct_predictions = 0
        
        print(f"Evaluating {total_samples} samples...\n")
        
        for index, row in tqdm(df.iterrows(), total=total_samples, desc="Evaluating", unit="query"):
            query = row['text']
            true_label = str(row['label_text']).strip()
            
            predicted_label = clean(classifier(query))
            
            if predicted_label == true_label:
                correct_predictions += 1
                
        accuracy = (correct_predictions / total_samples) * 100
        
        print("\n\n" + "="*50)
        print("                RESULTS                ")
        print("="*50)
        print(f" - Test sample size: {total_samples}")
        print(f" - Correct predictions: {correct_predictions}")
        print(f" - Accuracy: {accuracy:.3f}%")
        print("="*50 + "\n")
        
    except FileNotFoundError:
        print(f"Error: Cannot find {test_file_path}.")