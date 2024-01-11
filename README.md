The directory "STGIC" contains all the scripts which are organized as a Python package. The AGC algorithm is implemented by the script AGC.py, the dilated convolution framework is implemented by the script DCF.py, and data preprocessing is implemented by preprocess.py. The directory "test" contains all the scripts to test the clustering performance of STGIC. DLPFC.py is for testing with the bechmark of 10x Visium human DLPFC dataset consisting of 12 samples. human_bc.py is for testing the 10x Visium breast cancer dataset. mouse_postbrain.py is for testing the 10x Visium mouse posterior brain dataset. mouse_olfactory.py is for testing the Stereo-seq mouse olfactory bulb dataset. mybin.py is for binning the stereo-seq mouse olfactory bulb data.run_STGIC.ipynb is the jupyter notebook script for STGIC testing on 12 DLPFC samples and contains the operation result. All experiments are executed on a Centos7.9.2009 server equipped with an NVIDIA A100 GPU (NVIDIA-SMI 530.30.02). The testing data can be downloaded from https://zenodo.org/records/10477149. 
These scripts can be operated with docker. the directive is as follows:
docker run -itd --gpus all --rm -p 8888:8888 --user=root  henryc101/stgic:0.1 jupyter lab \
--no-browser --ip=0.0.0.0 --port=8888  --allow-root --ServerApp.allow_origin='*' --notebook-dir='/workspace' \
--ServerApp.token='36c69bc677bfd4e6d162946df346d63a1cdac4792ebff592'
