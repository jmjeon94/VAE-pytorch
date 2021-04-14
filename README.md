# Implementation of VAE
This is implementation of VAE(Variational Auto Encoder).

## Usage
### How to Train
Train Data: MNIST
```shell script
$ python3 train.py
```
### How to show results
```shell script
$ python3 evaluate.py
```

## Results 
Below is 50 epoch training result.
### 2D Feature Distribution
![distribution](./resource/distribution.png)
### Output of Normal Distribution (N(0,1))
![outputs](./resource/outputs.png)

### Reference
paper link: https://arxiv.org/abs/1312.6114
