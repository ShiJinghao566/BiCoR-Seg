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

Download datasets:
- LoveDA: https://codalab.lisn.upsaclay.fr/competitions/421
- ISPRS Vaihingen & Potsdam: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx


Data Preprocessing
------------------
**Vaihingen**
python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512

python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded

python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt


**Potsdam**
python tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image

python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image

python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image


**LoveDA**
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert


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
