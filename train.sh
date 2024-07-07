#!/bin/bash

pip install virtualenv
virtualenv -p python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
pip install boto3 --upgrade
pip install text-unidecode
python -m pip install git+https://github.com/NVIDIA/NeMo.git@r2.0.0rc0#egg=nemo_toolkit[asr]
sudo apt-get install -y sox libsndfile1 ffmpeg
pip install "numpy<2.0"
python train.py  --train_csv $1 \
                 --train_data_path $2 \
                 --adapt_csv $3 \
                 --adapt_data_path $4