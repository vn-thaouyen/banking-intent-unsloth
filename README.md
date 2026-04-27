# Banking Intent Classification with Llama 3.1 & Unsloth

An end-to-end pipeline for fine-tuning and evaluating **Llama 3.1 8B** on the **BANKING77** dataset (3000 samples). This project uses **Unsloth** and **LoRA (Low-Rank Adaptation)** to achieve highly efficient, memory-optimized banking intent classification for customer service queries.

## Key Features

* **Efficient Fine-Tuning:** Utilizes `unsloth` to train 2x faster while consuming significantly less VRAM compared to standard Hugging Face implementations.
* **4-bit Quantization:** Integrates `bitsandbytes` to load the base model in 4-bit precision, enabling the training of an 8B parameter model on a single consumer-grade GPU (e.g., NVIDIA T4/L4).
* **Modular Architecture:** Clean separation of configurations (`.yaml`), scripts (`.py`), and execution commands (`.sh`) following industry standards.
* **Automated Evaluation:** Built-in inference script to evaluate model accuracy against test datasets using a structured progress pipeline.

## Repository Structure

```text
banking-intent-unsloth/
├── configs/
│   ├── train.yaml           # Hyperparameters for SFTTrainer and LoRA config
│   └── inference.yaml       # Configuration for loading adapters and inference
├── scripts/
│   ├── inference.ipynb      # Notebook for quick inference        
|   ├── inference.py         # Performs single predictions and computes accuracy on the test set
|   ├── preprocess_data.py   # Downloads dataset, formats prompts, and exports to .csv file
|   ├── train.ipynb          # Notebook for training and monitoring VRAM usage interactively
|   └── train.py             # Main script for training the model   
├── sample_data/
│   ├── train.csv            # Pre-formatted training dataset
│   └── test.csv             # Test dataset
├── train.sh                 # Executable script to trigger training
├── inference.sh             # Executable script to trigger inference/evaluation
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Getting Started

Follow these step-by-step instructions to set up the environment, download the necessary assets, train the model, and run inferences. 

### 1. Set up the Environment
This project is highly optimized for NVIDIA GPUs (Tesla T4, L4, A100) and is best executed in a Linux/Google Colab environment.

**Step 1.1: Clone the repository**
```bash
git clone https://github.com/vn-thaouyen/banking-intent-unsloth.git
cd banking-intent-unsloth
```

**Step 1.2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. Download Model and Dataset

The pipeline is designed to automatically download the base model from Hugging Face and load the train dataset from sampla_data/train.csv.

### 3. Train the Model

The training process uses LoRA adapters to fine-tune the Llama 3.1 8B model. All hyperparameters (model name, lora, training arguments, model output directory, etc.) can be modified in `configs/train.yaml`.

Run the training script:

```bash
chmod +x train.sh
./train.sh
```

### 4. Run Model (Inference & Evaluation)

Once the model is trained, you can run the inference script. This script performs two tasks:

1. Predicts the intent for a single sample query to demonstrate functionality  
2. Evaluates the model's accuracy against the test dataset`sample_data/test.csv`

Run the inference script:

```bash
chmod +x inference.sh
./inference.sh
```

## Hyperparameters

The following hyperparameters are managed in `configs/train.yaml`. These values can be tuned for stability and performance.

| Parameter              | Value          |
|------------------------|----------------|
| Batch Size             | 2 (per device) |
| Learning Rate          | 2e-4           |
| Logging Step           | 1              |
| Optimizer              | adamw_8bit     |
| Epochs                 | 2              |
| Max Sequence Length    | 2048           |
| Warmup Ratio           | 0.05           |
| Weight Decay           | 0.01           |
| Gradient Accumulation  | 8              |
| LR Scheduler Type      | linear         |
| LoRA Dropout           | 0              |
| LoRA Rank              | 32             |
| LoRA Alpha             | 64             |
