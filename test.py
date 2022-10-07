import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable

transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])


transform_test = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True,num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data/', train=False, 
                                        download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                        shuffle=False, num_workers=2, drop_last=True)



def get_net():
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')
    net = models.resnet18(pretrained=True)
    #net_fit = net.fc.in_features
    net.fc=torch.nn.Linear(512,10)
    #net = nn.Sequential(nn.Linear(net_fit, 10), nn.Softmax(dim=1)) 
    return net

#devices=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

net = get_net()
#print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3,weight_decay=5e-4) #weight_decay为权重衰减  amsgrad=True
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,gamma=0.9) #优化lr,每隔 lr_period个epoch就给当前的lr乘以lr_decay

def accuracy(pred,target):
    pred_label=torch.argmax(pred,1)  #返回每张照片10个预测值里面最大数的标签
    correct=sum(pred_label==target).to(torch.float ) #如果这个标签等于真实标签，则将数值转化为个数，转化为float类型并返回给correct
    return correct,len(pred)#返回正确的个数

acc={'train':[],"val":[]}
loss_all={'train':[],"val":[]}

celoss=nn.CrossEntropyLoss(reduction="none")
best_acc=0

'''def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()), len(y_hat)
'''
def train_batch_ch13(net, X, y, celoss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需（稍后讨论）
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = celoss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum, train_prednum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum, train_prednum

"""训练和验证"""
net.to(devices[0])
for epoch in range(10):
    print('epoch',epoch+1,'*******************************')
    
    net.train()
    train_total_loss,train_correctnum, train_prednum=0.,0.,0.
    for images,labels in trainloader:
        l, acc, prednum = train_batch_ch13(net, images, labels, celoss, optimizer, devices)
        '''images,labels=images.to(devices),labels.to(devices)      
        outputs=net(images)
        loss=celoss(outputs,labels)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        train_total_loss =loss.sum()'''
        
        #correctnum, prednum= accuracy(outputs,labels)
        train_total_loss += l
        train_correctnum += acc
        train_prednum += prednum
        
       
    net.eval()
    valid_total_loss,valid_correctnum, valid_prednum=0.,0.,0.
    for images,labels in testloader:
        images,labels=images.to(devices[0]),labels.to(devices[0]) 
        images = Variable(images,requires_grad=True)
        #labels = Variable(labels,requires_grad=True)
        with torch.no_grad():  
            outputs=net(images)
        loss=celoss(outputs,labels)
        valid_total_loss += loss.sum()
        
        correctnum,prednum= accuracy(outputs,labels)
        valid_correctnum +=correctnum
        valid_prednum +=prednum
    scheduler.step()
    """计算平均损失"""
    train_loss = train_total_loss/len(trainloader) 
    valid_loss = valid_total_loss/len(testloader)
    
    """将损失值存入字典"""
    loss_all['train'].append(train_loss ) 
    loss_all['val'].append(valid_loss)
    """将准确率存入字典"""
    #acc['train'].append(train_correctnum/train_prednum)
    #acc['val'].append(valid_correctnum/valid_prednum)
#     lr.append(lr)
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
    print('Training Acc: {:.6f} \tValidation Acc: {:.6f}'.format(train_correctnum/train_prednum,valid_correctnum/valid_prednum))