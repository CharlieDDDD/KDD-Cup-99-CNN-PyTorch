# KDD Cup 99 DNN PyTorch

**note**: 1. Some code is inherited from others. ~~2. The project is still under development.~~

​	This is a classification model with five classes (normal, DOS, R2L, U2R,PROBING).  Ignore the content features of TCP connection ( columns 10-22 of KDD Cup 99 dataset)  when training the model to adapt [this project that a kdd99 feature extractor](https://github.com/AI-IDS/kdd99_feature_extractor).

## Network Structure

(Conv2d => ReLU )*2   => ( MaxPool2d )=> (Linear => [dropout] => ReLU) * 2 => ( Linear )

## Requirements

* PyTorch 1.4+

<h2 id="Performance">To do list</h2>

- [ ] ~~optimize data initialization (standardization，deal error data, etc)~~
- [ ] ~~predict.py~~
- [ ] ~~more evaluation methods (Confusion Matrix, Recall, etc)~~

## Data Preparation

​	**First**,  unzip the data (

​			**Train_data:** kddcup.data.gz,

​			**Train_data_10%**: kddcup.data_10_percent.gz,

​			**Test_data**:  corrected.gz )

​		in the **/data_pre_processing** folder. or download the data from [the official Website ](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)  

​	**Second**,  convert strings in the datasets to discrete numbers

```python
#using 10% training data
python data_pre_processing_10%.py
# or, using all training data
python data_pre_processing_all.py
```

**Third**, copy train_data* and test.csv to the ./dataset, and change the data path in train.py file


## Training

```bash
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD]
Train the DNN on KDD Cup 1999. Note: the default parameters are not the
best!!!
optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 512)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
```

**e.g.** (It's working)

```shell script
python train.py -e 20 -b 512 -l 0.0001
```

## Tensorboard

Visualize the train and test losses, accuracy,  the weights and gradients in real time.

```shell
tensorboard --logdir=runs
```

<h2 id="Performance">Performance</h2>

| training_dataset | accuracy |
| :--------------: | :------: |
|       10%        |  0.9395  |
|       All        |  0.9393  |

## Test

~~To be done.~~

## Predict 

~~To be done.~~

### Reference

~~To be done.~~