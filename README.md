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
 pip3 install torch torchvision
 pip3 install gym==0.15.3
```