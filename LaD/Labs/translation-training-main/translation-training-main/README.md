# Translation with JoeyNMT

This tutorial is based on the [Joey NMT v2 Quick Start Tutorial](https://github.com/joeynmt/joeynmt/blob/main/notebooks/joey_v2_demo.ipynb). We use here the smaller [Opus Salome corpus](https://opus.nlpl.eu/Salome/de&en/v1/Salome) of [Poncelas et al. (2020)](https://aclanthology.org/2020.loresmt-1.14/) and a smaller transformer model to speed up training.


## Setup

Prepare a kernel with joeynmt as follows

1. Create a python environment (with virtualenv, venv, conda, etc.), preferably with a recent python version
```
virtualenv --python="/usr/bin/python3.12" .venv12    
source .venv12/bin/activate
```
2. Install joeynmt, jupyter and ipykernel
```
pip install git+https://github.com/joeynmt/joeynmt.git
pip install jupyter ipykernel
```
3. Define a kernel based on your python environment
```
python -m ipykernel install --user --name .venv12
```

You can now launch the notebook (`joeynmt_tutorial.ipynb`) and select the `.venv12` kernel from your notebook/lab. 