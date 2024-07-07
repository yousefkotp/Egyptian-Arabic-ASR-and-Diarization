#!/bin/bash

apt-get install -y sox libsndfile1 ffmpeg
pip install text-unidecode


## Install NeMo
BRANCH='r2.0.0rc0'
python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

pip install -r requirements.txt