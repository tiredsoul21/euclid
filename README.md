# euclid

## Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip3 install numpy==1.26.3
pip3 install gym==0.26.2
pip3 install torch==2.1.2
pip3 install pytorch-ignite==0.4.13
pip3 install tensorboard==2.15.1
pip3 install matplotlib==3.8.2
```

## Monitor
To see output visualizations:
`tensorboard --logdir runs` 

## Dataset
https://www.kaggle.com/datasets/tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update

## Execution
```
python3 src/00_example.py -p /home/derrick/data/daily_price_data -r test --cuda
python3 src/run_model.py -d /home/derrick/data/daily_price_data/test/FOX.csv -m output/00-test/valReward-13.596.data -n test1
```