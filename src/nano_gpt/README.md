# euclid / Nano GPT
## Nano GPT Model
### Notes
This was primarily derived from: <br>
https://github.com/karpathy/nanoGPT/tree/master <br>
and the Githubs repos listed within.  The implementation and model designs are not unique and not owned / created by me.
### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip3 install numpy==1.26.3            # For data management
pip3 install torch==2.1.2             # Core models / networks
pip3 install tiktoken==0.6.0          # Text tokenization
pip3 install datasets==2.17.1         # For dataset managment
```

### Dataset
https://huggingface.co/datasets/Skylion007/openwebtext

### Execution
```
# Help:
python3 -m src.nano_gpt.train --help

# Train model from scratch
python3 -m src.nano_gpt.train
# Runs about 10% faster:
python3 -m src.nano_gpt.train --compile -d /home/derrick/data/openwebtext

# Evaluate the model only
python3 -m src.nano_gpt.train --eval_only --resume_run -o /home/derrick/repo/euclid/trial1 -i 1000 -d /home/derrick/data/openwebtext

# Resume training of an existing model
# -i Iteration (written into name)
# -o folder containing model stuffs
python3 -m src.nano_gpt.train --resume_run -o /home/derrick/repo/euclid/trial1 -i 1000 -d /home/derrick/data/openwebtext
# Runs about 10% faster:
python3 -m src.nano_gpt.train --resume_run -o /home/derrick/repo/euclid/trial1 -i 6000 --compile -d /home/derrick/data/openwebtext

# Query to model
python3 -m src.nano_gpt.train --generate -m /home/derrick/repo/euclid/test/ckpt_6000.pt -q "Where do you live?"
```

### Runs
# single dataset
python3 -m src.nano_gpt.train -d /home/derrick/data/openwebtext
python3 -m src.nano_gpt.train --resume_run -o /home/derrick/repo/euclid/trial1 -i 1000 --compile -d /home/derrick/data/openwebtext

# Multiple Datasets
python3 -m src.nano_gpt.train --resume_run -o /home/derrick/repo/euclid/trial1 -i 46000 --compile -d /home/derrick/data/openwebtext,/home/derrick/data/python
