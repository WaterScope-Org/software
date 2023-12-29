# !/bin/bash

sudo apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install --upgrade setuptools 
sudo -H pip3 install Cython
wget https://github.com/Kashu7100/pytorch-armv7l/raw/main/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
sudo -H pip3 install --no-cache-dir torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl 
sudo -H pip3 install https://github.com/Kashu7100/pytorch-armv7l/raw/main/torchvision-0.8.0a0%2B45f960c-cp37-cp37m-linux_armv7l.whl
sudo -H pip3 install tqdm
sudo -H pip3 install gdown

# gdown https://drive.google.com/uc?id=1U-vQLZZtK54cgNhLShSm-ie-H4sDMYGr -O inference_script/
# gdown https://drive.google.com/uc?id=1hjHyNFn1jxw79vNTiS7AAHoobFiwBxn4 -O inference_script/
gdown https://drive.google.com/uc?id=1xhtuQjLLGLcawD93Eyia7AHSf1461nzJ -O inference_script/
