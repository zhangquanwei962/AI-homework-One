# CIFAR10

## 1.Introduction

这是人工智能安全的课程作业的第一课，用cifar10为数据集训练分类  
这个数据集是只有10个类别的  
  
采用`pytorch`提供的预训练好的`ResNet18`，（`ResNet101`把服务器显卡都跑挂了）  

**如果是只有单个GPU的话，请运行main.py**；  
  

**如果是多个GPU的话，请运行`main1.py`获取resnet18， 运行`main-resnet34.py`获取resnet34;**  
  
    
___

请保证虚拟环境中有最新版的python（>3.7),pytorch(2022.10)
  
## 2.Preparation

> 请把文件直接clone下来，请使用：  
> `git clone https://github.com/zhangquanwei962/AI-homework-One.git` 

保证文件结构如下：
> AI-homework-One
>> data  
>>> cifar-10-python.tar.gz  
>>> cifar-10-batches-py  

>> main1.py  
>> main2.py  
>> main-resnet34.py

**请注意，如果`data`下面没有数据集的话，请把`trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,download=False, transform=transform_train)`和`testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)`里面的`download`改成`True`**  
  
  ___
## 3.Run
一切准备就绪，就直接在你所有的虚拟环境运行即可  
`python3 main1.py`，等待程序结束
## hyper parameter
batch_size, epoch, lr, weigth_decy, lr_period, lr_decay
## 5.Issue
如果有问题，请提issue，看到会处理

