# euclid


## Character GPT Model
### Notes
This was primarily derived from: <br>
https://karpathy.ai/zero-to-hero.html <br>
and the corresponding Juniper notebooks / Githubs repos listed within.  The implementation and model designs are not unique and not owned / created by me.
### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip3 install torch==2.1.2             # Core models / networks
pip3 install matplotlib==3.8.2        # For plotting
pip3 install torchviz==0.0.2          # Graph visualization
```

### Dataset
https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
https://github.com/karpathy/makemore/blob/master/names.txt

### Execution
```
python3 -m src.char_gpt.train  -p data/tinyshakespear.txt
```
