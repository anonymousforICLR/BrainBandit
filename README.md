This is the official repository for anonymous submission for ICLR2025 named **Brain Bandit: A Biologically Grounded Neural Network for Efficient Control of Exploration**

The MDP part is built on the work from Ian Osband https://github.com/iosband/TabulaRL

The mice behavior part is built on the work from https://github.com/celiaberon/2ABT_behavior_models

We would like to express our sincere thanks for them and for other works that inspired our research. 

## Environment
python==3.8

numpy==1.20.3

pandas==2.0.3

tqdm==4.65.0

scipy==1.10.1

matplotlib==3.7.2

sympy==1.12

statsmodels==0.14.0

We run all the experiments on Ubuntu 20.04 with an i9-13900K(CPU) and a Rtx 4090(GPU).

If you get an error about numpy, it is likely that the numpy version is too high, so please use the provided version.

If you cannot run correctly with multiprocessing on Windows using jupyter notebook, please copy the code to a .py file and run the python file.  