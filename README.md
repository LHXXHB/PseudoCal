# Code for **PseudoCal**@ICML 2024

## [**Pseudo-Calibration: Improving Predictive Uncertainty Estimation in Unsupervised Domain Adaptation**](https://openreview.net/forum?id=XnsI1HKAKC)

### Prerequisites
- python == 3.7.13 
- cudatoolkit == 10.1.243
- pytorch ==1.7.1
- torchvision == 0.8.2
- numpy, scikit-learn, PIL, argparse

### Demo

- Configure the PyTorch environment.
- Download the [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) dataset. Configure the data lists in **data** and the checkpoints in **logs**.
- Run the code in **pseudocal.sh**.


### Citation

> @inproceedings{hu2024pseudocalibration,  
> &nbsp; &nbsp;  title={Pseudo-Calibration: Improving Predictive Uncertainty Estimation in Unsupervised Domain Adaptation},  
> &nbsp; &nbsp;  author={Dapeng Hu and Jian Liang and Xinchao Wang and Chuan-Sheng Foo},  
> &nbsp; &nbsp;  booktitle={Forty-first International Conference on Machine Learning},   
> &nbsp; &nbsp;  year={2024}  
> }


### Contact

- [lhxxhb15@gmail.com](lhxxhb15@gmail.com)


### Credit
- The code is heavily borrowed from [TransCal](https://github.com/thuml/TransCal).
