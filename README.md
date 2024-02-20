The directory "STGIC" contains all the scripts which are organized as a Python package.<br><br>
The AGC algorithm is implemented by the script AGC.py, the dilated convolution framework is implemented by the script DCF.py, and data preprocessing is implemented by preprocess.py. The directory "test" contains all the scripts to test the clustering performance of STGIC. DLPFC.py is for testing with the bechmark of 10x Visium human DLPFC dataset consisting of 12 samples. human_bc.py is for testing the 10x Visium breast cancer dataset. mouse_postbrain.py is for testing the 10x Visium mouse posterior brain dataset. mouse_olfactory.py is for testing the Stereo-seq mouse olfactory bulb dataset. mybin.py is for binning the stereo-seq mouse olfactory bulb data. <br><br>
run_DLPFCs.ipynb is the jupyter notebook script for STGIC testing on 12 DLPFC samples and contains the operation result. <br><br>
tutorial.ipynb is a detailed tutorial to instruct the usage of STGIC on 151507 in the benchmarkdataset and all non-benchmark datasets in the study, the operation results are also contained.<br><br>
package_install.sh is a shell script to install all the packages of the designated version.<br><br>
All experiments are executed on a Centos7.9.2009 server equipped with an NVIDIA A100 GPU (NVIDIA-SMI 530.30.02). The testing data can be downloaded from https://zenodo.org/records/10477149. <br><br>
These scripts can be operated with docker. The directive is as follows: docker run -it --gpus all --rm --user=root -v $(pwd):/workspace  henryc101/stgic:0.1 python run_STGIC_dlpfc_docker.py  <br><br> 
The above directive is for testing the 12 DLPFCs benchmark with the above docker image and NVIDIA GPU A100.
