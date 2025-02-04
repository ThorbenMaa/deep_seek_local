# Summary
Simple code to download one of the destilled DeepSeek-R1 models and host the locally on a laptot in a gradio webapp (using CPU).
GPU can also be utilized. If you have GPU available, adopt the installation of pytorch libraries as explained below.
# Setup
## install mamba
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
mamba init
```
## vreate env and install libraries
```
mamba create -n deep_seek
mamba activate depp_seek
mamba install pip
# install torch (see https://pytorch.org/get-started/locally/ to adjust the command to your need, e.g., if you want to use GPU)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install transformers tokenizers gradio
```
## run code and host model in web app 
```
python host_deep_seek_locally.py 
```

# Recommendations
If you use the QWEN 1.5b parameter model, set the tempertaure to 1.
Since its running on CPU, its pretty small. Play around with the pytorch installation and the
[`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) class 
to optimize the performance.