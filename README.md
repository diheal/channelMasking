# An Imporved Masking Strategy for Self-supervised Masked Reconstruction in Human Activity Recognition
This repository is designed to implement the idea of "An Imporved Masking Strategy for Self-supervised Masked Reconstruction in Human Activity Recognition" in this paper.

## Requirements

This project code is done in Python 3.8 and third party libraries. 

 TensorFlow 2.x is used as a deep learning framework.

The main third-party libraries used and the corresponding versions are as follows:

+ tensorflow 2.3.1

+ tensorflow_addons 0.15.0

+ numpy 1.18.5

+ scikit-learn 0.23.1


## Running

This demo can be run with the following command:

```shell
python main.py
```


## Code Organisation

The main content of each file is marked as follows:

+ [`dataset.py`](./dataset.py): Dataset processing
+ [`encoder.py`](./encoder.py): Model structure
+ [`encoderLayer.py`](./encoder.py): Model structure
+ [`main.py`](./main.py): Main Program Entry
+ [`module.py`](./module.py): Components needed to train the model
+ [`multiHeadAttention.py`](./multiHeadAttention.py): Model structure
+ [`utils.py`](utils.py): The tools and methods needed to train the model


## Citation

If you find our paper useful or use the code available in this repository in your research, please consider citing our work:

```

```

## Reference

+ https://github.com/dapowan/LIMU-BERT-Public
