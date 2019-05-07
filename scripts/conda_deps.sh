set -ex
conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9
conda install pybind11 opencv matplotlib -c conda-forge
conda install scikit-learn h5py
