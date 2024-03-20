# euclid

## Stock Model - DDQN
### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip3 install numpy==1.26.3            # For data management
pip3 install gym==0.26.2              # Utility to build / train RL 
pip3 install torch==2.1.2             # Core models / networks
pip3 install pytorch-ignite==0.4.13   # Engines
pip3 install tensorboard==2.15.1      # Results display
pip3 install matplotlib==3.8.2        # For plotting
pip3 install scipy==1.12.0            # For statistics
pip3 install line_profiler==4.1.2     # For performance profiling
pip3 install torchviz==0.0.2          # Graph visualization
```

### Monitor
To see output visualizations:
`tensorboard --logdir runs` 

### Dataset
https://www.kaggle.com/datasets/tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update

### Execution
```
# Training:
python3 src/trainStockModel.py -p /home/derrick/data/daily_price_data -r test --cuda

# Short training runs:
python3 src/trainStockModel.py -p /home/derrick/data/daily_price_data/other -r test --cuda

# Batch testing
# python3 src/testStockModel.py -d /home/derrick/data/daily_price_data/test/ -m valReward-22.174.data -n test-01

# Short testing run
# python3 src/testStockModel.py -d /home/derrick/data/daily_price_data/test/FOX.csv -m valReward-22.174.data -n test-01

# Create plots
# python3 src/testStockModel.py -d results-test-01.json -o "plot" -n test1
```

### Runs
```
  test-00 :: EPS_STEPS 1M || BATCH_SIZE 32  || 50 Bars
  test-01 :: EPS_STEPS 2M || BATCH_SIZE 64  || 50 Bars
  test-02 :: EPS_STEPS 1M || BATCH_SIZE 128 || 50 Bars
* test-03 :: EPS_STEPS 1M || BATCH_SIZE 64  || 50 Bars
------------------------------------------------------------------------------------
  test-04 :: EPS_STEPS 1M || BATCH_SIZE 64  || 100 Bars
  test-05 :: EPS_STEPS 1M || BATCH_SIZE 64  || 25 Bars 
* test-06 :: EPS_STEPS 1M || BATCH_SIZE 64  || 50 Bars -- Optimizations
------------------------------------------------------------------------------------
  test-07 :: Test against modularized methods - prep for additional feature extraction
  test-08 :: Prioritized Buffer test.  Did not seem impactful yet
* test-09 :: Test with 2D CNN for 3xN price data
  test-10 :: Test with long run and LEARNING_RATE 0.0001 --> 0.001
```

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
python3 -m src.char_gpt_model  -p data/tinyshakespear.txt


## Code Model
### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip3 install datasets==2.17.1         # For dataset managment
```