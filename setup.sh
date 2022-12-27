#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pytorch-sparse -c pyg
pip3 install pytorch_lightning yacs
pip3 install ogb
pip3 install optuna
pip3 install networkx
pip3 install torch-sparse

#conda install -c nvidia -c rapidsai -c numba -c conda-forge cugraph
#pip install cugraph-cu11 --extra-index-url=https://pypi.ngc.nvidia.com

pip install graphgym