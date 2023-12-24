# euclid

## Setup
```
# Code snippets
mkdir dependencies
cd dependencies
git clone https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition.git
mv Deep-Reinforcement-Learning-Hands-On-Second-Edition packt
cd packt
git checkout d5a421d63c6d3ebbdfa54537fa5ce485bc2b9220

sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip3 install torch==2.0.1
pip3 install torchvision==0.15.0
pip3 install gym==0.15.3
pip3 install tensorboardX
pip3 install opencv-python
pip3 install atari-py==0.2.6
pip3 install pytorch-ignite==0.4.13
```

## Monitor
To see output visualizations:
`tensorboard --logdir runs` 