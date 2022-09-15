Use a conda environment with pytorch 1.12.
```
conda create --name pt112 python=3.9 -y
conda activate pt112
conda install ipykernel jupyter -y
python -m ipykernel install --user --name pt112 --display-name "pt112"
```

Get pytorch installed. Command generated here: https://pytorch.org/
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge -y
```

Requirements
```
pip install -r requirements.txt
```
