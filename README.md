BiCoR

Installation
------------
conda create -n bicor python=3.8
conda activate bicor
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt


Dataset Preparation
-------------------
Organize the repository as follows:

BiCoR
├── network
├── config
├── tools
├── model_weights
├── fig_results
├── lightning_logs
├── data
│   ├── LoveDA
│   ├── vaihingen
│   ├── potsdam
│   ├── uavid


Training
--------
python train.py -c config/loveda/bicor.py
python train.py -c config/vaihingen/bicor.py
python train.py -c config/potsdam/bicor.py


Testing
-------
python test_vaihingen.py -c config/vaihingen/bicor.py -o fig_results/vaihingen/bicor --rgb -t 'd4'
python test_potsdam.py -c config/potsdam/bicor.py -o fig_results/potsdam/bicor --rgb -t 'lr'
python test_loveda.py -c config/loveda/bicor.py -o fig_results/loveda/bicor --rgb -t "d4"
