# Image-specific information suppression and implicit local alignment for text-based person search

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/taksau/GPS-Net/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.7.1-%237732a8) 

We provide the code for reproducing result of our  paper [**Image-specific information suppression and implicit local alignment for text-based person search**](https://arxiv.org/pdf/2208.14365.pdf). 

## Getting Started
#### Dataset Preparation

1. **CUHK-PEDES**

   Organize them in `dataset` folder as follows:
       

   ~~~
   |-- dataset/
   |   |-- <CUHK-PEDES>/
   |       |-- imgs
               |-- cam_a
               |-- cam_b
               |-- ...
   |       |-- reid_raw.json
   
   ~~~

   Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) and then run the `process_CUHK_data.py` as follow:

   ~~~
   cd MANet
   python ./dataset/process_CUHK_data.py
   ~~~

2. **ICFG-PEDES**

   Organize them in `dataset` folder as follows:

   ~~~
   |-- dataset/
   |   |-- <ICFG-PEDES>/
   |       |-- imgs
               |-- test
               |-- train 
   |       |-- ICFG_PEDES.json
   
   ~~~

   Please request the ICFG-PEDES database from [chxding@scut.edu.cn](mailto:chxding@scut.edu.cn) and then run the `process_ICFG_data.py` as follow:

   ~~~
   cd MANet
   python ./dataset/process_ICFG_data.py
   ~~~

#### Training and Testing
~~~
python train.py 
~~~
#### Evaluation
~~~
python test.py 
~~~

### Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate for their contributions.

* [SSAN](https://github.com/zifyloo/SSAN)

## Citation

If this work is helpful for your research, please cite our work:

~~~
@article{MANet,
   title={Image-Specific Information Suppression and Implicit Local Alignment for Text-based Person Search}, 
   author={Shuanglin Yan and Hao Tang and Liyan Zhang and Jinhui Tang},
   journal={IEEE Transactions on Neural Networks and Learning Systems}, 
   year={2023},
   volume={},
   number={},
   pages={1-14},
   doi={10.1109/TNNLS.2023.3310118}
}
~~~
