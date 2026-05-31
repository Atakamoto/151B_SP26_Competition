# CSE 151B Competition — Starter Code

# GPU: 
A100 via Google Colab 
# Inference Time:
~6 hours
# How to run
We ran it using a jupyter notebook in google colab with the following commands. Could run in terminal but we need to run in notebook to stop runtime from disconnecting in google colab.
```
!git clone https://github.com/Atakamoto/151B_SP26_Competition.git
```
```
!pip uninstall -y vllm torch torchvision torchaudio xformers
!pip install -U uv
!uv pip install --system --no-cache vllm==0.19.1
!uv pip install --system --no-cache transformers==4.57.6 tqdm sympy numpy bitsandbytes antlr4-python3-runtime==4.11.1 accelerate
```

```
from google.colab import drive
drive.mount('/content/drive')
```
(not necessary just so we could store results before runtime disconnect deletes files)
```
%cd /content/151B_SP26_Competition
```
(for google colab specifically)

```
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_csv="/content/drive/MyDrive/cse151b_submission.csv",
    debug_csv="/content/drive/MyDrive/cse151b_debug.csv", 
)
```
(debug csv not relevant)
