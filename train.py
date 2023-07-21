import torch
import argparse
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--base-model', type=str, help='Set Base Model')
parser.add_argument('--dataset', type=str, help='Set Data Path')
parser.add_argument('--output', type=str, help='Set the output model path')
parser.add_argument('--epochs', type=int, help='Set the number of epochs')
parser.add_argument('--steps', type=int, help='Set the number of steps')
parser.add_argument('--int8', action='store_true', help='Enable int8 quantization')
args = parser.parse_args()

if args.base_model:
    BASE_MODEL = args.base_model
    print(f"Using model: BASE_MODEL")
else:
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"
    print("No model provided, using default: meta-llama/Llama-2-7b-hf")
if args.dataset:
    DATA_PATH = args.dataset
    print(f"Using dataset: {DATA_PATH}")
else:
    DATA_PATH = "tatsu-lab/alpaca"
    print("No data path provided, using tatsu-lab/alpaca")
if args.output:
    OUTPUT_PATH = args.output
    print(f"Using output path: {OUTPUT_PATH}")
else:
    OUTPUT_PATH = "mymodel-finetuned"
    print("No output path provided, defaulting to mymodel-finetuned")
if args.epochs:
    EPOCHS = args.epochs
    print(f"Epochs: {EPOCHS}")
else:
    EPOCHS = 3
    print("No epochs count provided, defaulting to 3")
if args.steps:
    STEP_COUNT = args.steps
    print(f"Learning Steps: {STEP_COUNT}")
else:
    STEP_COUNT = 10000
    print("No step count provided, defaulting to 10k")
if args.int8:
    USE_INT8 = True
    print(f"Using int8 quantization")
else:
    USE_INT8 = False
    print("Not using int8 quantization")

MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-4
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_FP16 = True
USE_BF16 = False

if torch.cuda.is_available():
    print('Torch & Cuda Detected')
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f'GPU Name [{i}]: ', torch.cuda.get_device_name(i))
    if torch.cuda.device_count() > 1:
        print("Using Automatic Device Mapping.")
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            load_in_8bit=USE_INT8,
        )
    else:
        print("Using Single GPU.")
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=USE_INT8,
        )

amp_supported = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

if amp_supported:
     print(f"AMP Supported: {amp_supported}")
     bfloat16_supported = torch.cuda.is_bf16_supported()
     print(f"BFLOAT16 Supported: {bfloat16_supported}")
     if bfloat16_supported:
          USE_FP16 = False
          USE_BF16 = True

tokenizer = LlamaTokenizer.from_pretrained(
    BASE_MODEL,
    model_max_length=CUTOFF_LEN,
    padding_side="right",
    use_fast=False,
)

tokenizer.add_special_tokens(
        {
            
            "pad_token": "<PAD>",
        }
    )

config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
        )
if USE_INT8:
    print("Preparing model for int8 training...")
    model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)


try:
        # Try to load the dataset from local directory
        data = load_dataset(DATA_PATH, download_mode='reuse_cache_if_exists')
except FileNotFoundError:
        # If not found locally, download the dataset from Hugging Face
        print(f"Dataset {DATA_PATH} not found locally. Downloading from Hugging Face...")
        data = load_dataset(DATA_PATH, download_mode='force_redownload')


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        padding="longest",
        max_length=CUTOFF_LEN,
        truncation=True,
    )
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        max_steps=STEP_COUNT,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=USE_FP16,
        bf16=USE_BF16,
        logging_steps=10,
        output_dir=OUTPUT_PATH,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_torch_fused",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)