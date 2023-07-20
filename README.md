# LLaMAV2-Alpaca-Toolkit  (WIP)

Training and inference code for [LLaMA 2](https://ai.meta.com/llama/) models based on the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) instruct format.
### Usage
```
python train.py --base-model huggingface/modelname --dataset huggingface/datasetname --epochs number of epochs to train --steps number of total steps --output trained model name
```
### Example
```
python train.py --base-model meta-llama/Llama-2-7b-hf --dataset "tatsu-lab/alpaca"" --output alpaca-v2-7b --epochs 3 --steps 10000
```

### Credits
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- [Manuel030/alpaca-opt](https://github.com/Manuel030/alpaca-opt)
- [Meta AI LLaMA 2](https://ai.meta.com/llama/)
- [facebookresearch/llama-recipes](https://github.com/facebookresearch/llama-recipes/)
