# Latent Diffeomorphic Dynamic Mode Decomposition

    [1] W. Diepeveen, J. Schwenk, A. Bertozzi.  
    Latent diffeomorphic dynamic mode decomposition
    arXiv preprint arXiv:xxxx.xxxx. 2025 MMM DD.

Setup
-----

The recommended (and tested) setup is based on MacOS 15.4 running Python 3.8. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name lddmd python=3.8
    conda activate lddmd

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Latent-Diffeomorphic-Dynamic-Mode-Decomposition.git
    cd "Latent-Diffeomorphic-Dynamic-Mode-Decomposition"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

To produce the results in [1]. 
* For the synthetic data results run `toy_lddmd.ipynb`.
* For the real data results run `basin_lddmd.ipynb`.
