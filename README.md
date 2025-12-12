# LoTa-Bench / LLMTaskPlanning – Setup Guide (Paperspace)

This README is for complete working setup for running **LoTa-Bench** (LLMTaskPlanning) on a **headless Paperspace instance**.  
It includes all required installations, fixes, configuration updates, and execution instructions.

---

## 1. Clone Repository
```bash
cd /notebooks
git clone https://github.com/lbaa2022/LLMTaskPlanning.git
cd LLMTaskPlanning
```

---

## 2. Create Virtual Environment (Persistent)
```bash
python3 -m venv lota-bench-venv
source lota-bench-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 3. Install PyTorch (Working Version)
```bash
pip install "torch==2.0.0" "torchvision==0.15.1"
```

---

## 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Download ALFRED Dataset
```bash
cd alfred/data
sh download_data.sh json
cd ../..
```

---

## 6. Install System Dependencies  
Fixes missing `lspci`, X11, fonts, OpenCV errors:

```bash
apt-get update
apt-get install -y pciutils xvfb fonts-ubuntu fonts-dejavu-core
```

---

## 7. Start Virtual Display
```bash
cd /notebooks/LLMTaskPlanning
source lota-bench-venv/bin/activate
Xvfb :1 -screen 0 1024x768x24 &
```

---

## 8. Fix Hydra Display Configuration  
Update this block in `conf/config_alfred.yaml`:

```yaml
alfred:
  x_display: "1"
```

AI2-THOR requires a string and not an integer which in case is there in the config file.  

---

## 9. Run ALFRED Evaluation
```bash
DISPLAY=:1 python src/evaluate.py --config-name=config_alfred
```

---

## 10. Running WAH (Watch-And-Help)
Qwen models do not work due to tokenizer + AutoModel incompatibility.  
Use GPT‑Neo instead.

Run:
```bash
./run_wah_eval.sh
```

Logs saved to:
```
logs/wah/
```


---

## Fixes and Improvement Applied

### ALFRED
- Replaced failing `startx.py` with `Xvfb`
- Installed fonts to fix PIL/OpenCV errors
- Ensured YAML uses `x_display: "1"`  
- Prevented Hydra overrides from converting string → int  

### WAH
- Removed Qwen-related args (`trust_remote_code`, tokenizer mismatches)
- Forced GPT‑Neo to avoid AutoModel errors
- Cleaned duplicated keyword arguments
- Ensured scoring_mode works in naive mode  


---
## Credits
- LoTa-Bench: https://github.com/lbaa2022/LLMTaskPlanning  
- ALFRED Dataset: https://github.com/askforalfred/alfred  
- WAH Dataset: Part of VirtualHome  

-Large datasets, models, and simulator assets are excluded due to size limits; logs, outputs, resources, and all experiment code are included.


---

