# DLCV-Fall-2022-HW2
Please click [this link](https://docs.google.com/presentation/d/1A38mJUAfDo-4yYzy6UCBZrEo3aE50ceO/edit?usp=sharing&ouid=107585355306558125830&rtpof=true&sd=true) to view the slides of HW2

## Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2022/hw2-<username>.git
Note that you should replace `<username>` with your own GitHub username.

## Submission Rules
### Deadline
2022/10/31 (Mon.) 23:59 (GMT+8)

### Packages
This homework should be done using python3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.
    
You can run the following command to install all the packages listed in the requirements.txt:

    conda create --name dlcv-hw2 python=3.8
    conda activate dlcv-hw2
    pip3 install --target=/home/zhongwei/.conda/envs/DLCV-hw2/bin/ -r requirements.txt

https://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
如果少了任何 module，自己手動 pip install 最快。

If you have 2 GPUs, please do the following manually: (the training is based on CUDA:1)

    export CUDA_VISIBLE_DEVICES=0,1

### List all environments

    conda info --envs
    
### Check all package environment

    conda list -n DLCV-hw2

### Close an environment

    conda deactivate

### Remove an environment

    conda env remove -n dlcv-hw2

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

# Train
## 2-1

    bash ./train2-1.sh ./output_images
    
# Eval
## 2-1

    bash ./eval2-1.sh ./output_images

# Q&A
If you have any problems related to HW2, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw2 FAQ section in FB group.(But TAs won't answer your question on FB.)

# Troubleshooting
<module> could not be resolved by pylance (in VS code): Select intepreter（在VS code右下角）

### pip installing in global site-packages instead of virtualenv.
https://stackoverflow.com/questions/20952797/pip-installing-in-global-site-packages-instead-of-virtualenv
#### Solution
See conda virtual environment package location.
https://stackoverflow.com/questions/35709497/anaconda-python-where-are-the-virtual-environments-stored
    source activate DLCV-hw2
    which python
output: /home/zhongwei/.conda/envs/DLCV-hw2/bin/python
    
 
### ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.    
https://stackoverflow.com/questions/40183108/python-packages-hash-not-matching-whilst-installing-using-pip

