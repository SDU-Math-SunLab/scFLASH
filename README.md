# scFLASH: Flexible muLti-sample single-cell data integrAtion with phenotypic expreSsion Heterogeneity #

## Introduction ##
`scFLASH` is a deep learning-based framework designed for integrating multi-batch, multi-condition (MBMC) single-cell sequencing datasets. By preserving biological and condition-specific signals while correcting technical noise, scFLASH provides a clearer understanding of complex single-cell data. The workflow of scFLASH is shown in the following Figure:

<p align="center">
<img src=Figure_Method.jpg width=700ptx>
</p>


## News ##
* Dec, 2024: scFLASH version 1.0.0 is launched.

## Python Dependencies
`scFLASH` depends on the following Python packages:
```{bash}
python >= 3.8
numpy >= 1.23.5
pandas >= 1.5.3
torch >= 1.12.1
scib >= 1.1.4
scanpy >= 1.9.8
```



## How to install
To install scFLASH, follow these steps:

### Step1:
Create a new conda environment (recommended):
```{bash}
conda create -n scFLASH python=3.8
conda activate scFLASH
```
### Step2:
Install PyTorch with the appropriate CUDA version (or CPU-only if CUDA is unavailable):<br>
* For CUDA 11.3:
```{bash}
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
* For other CUDA versions or CPU-only installation: Visit [PyTorch’s official installation page](https://pytorch.org/get-started/locally/) and select the command that matches your setup.

### Step3:
Clone the scFLASH repository and install:
```{bash}
git clone https://github.com/SDU-Math-SunLab/scFLASH
cd scFLASH
pip install .
```
## Quick start
```python
import scFLASH

# prepare data source
input_dict = scFLASH.prepare_anndata(adata, batch_key="batch", condition_key="condition")

# init a scFLASH model
scflash_model = scFLASH.Integrator(input_dict, lamda_batch=None, cond_factor_k = 0.5, device = 'cuda:0')

# run
scflash_model.fit(batch_size=300, num_epochs=100,lr=None, mu0=0.001)

# get the integrated result
adata_int = scflash_model.get_corrected_exp(input_dict)
```

Please see the [tutorial.ipynb](https://github.com/SDU-Math-SunLab/scFLASH/blob/main/tutorial/tutorial.ipynb) for a comprehensive workflow on integrating datasets with scFLASH. This tutorial uses the Alzheimer’s Disease (AD) dataset as an example to demonstrate how scFLASH can be applied in real applications.


## How to cite `scFLASH` ##
Please cite the following manuscript:

> *Integrating multi-sample single-cell data with phenotypic expression heterogeneity*. 
<br />


## License ##
scFLASH is licensed under the GNU General Public License v3.0.

Improvements and new features of scFLASH will be updated on a regular basis. Please post on the [GitHub issues page](https://github.com/SDU-Math-SunLab/scFLASH/issues) with any questions.


