import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

import argparse
import time

from model import ColorizationNet
from dataset import BlackColorImages


# KMeans初始化
def kmeans_init(model, data, layer_name='fc1', n_clusters=10):
    layer = getattr(model, layer_name)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    
    # 将KMeans的中心点作为层的权重
    with torch.no_grad():
        layer.weight.data = centroids_tensor
        # 这里假设偏置项为0
        if layer.bias is not None:
            layer.bias.data.zero_()
            

class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial''' 
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
            
            
def train(train_loader, model, criterion, optimizer, epoch_idx, summarywriter=None, device='cuda'):
    print('Starting training epoches {}'.format(epoch_idx))
    model.train()
    
    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab) in enumerate(train_loader):
        
        # Use GPU if available
        if device == 'cuda': 
            input_gray, input_ab = input_gray.cuda(), input_ab.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('epocheses: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch_idx, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses)) 
            summarywriter.add_scalar('Loss/train', losses.avg, epoch_idx*len(train_loader)+i)

    print('Finished training epoches {}'.format(epoch_idx))


def validate(val_loader, model, criterion, save_images=False, device='cuda'):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if device == 'cuda': 
            input_gray, input_ab = input_gray.cuda(), input_ab.cuda()

        # Run model and record loss
        output_ab = model(input_gray) # throw away class predictions
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))


        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))
            

            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            image = to_pil(output_ab[0])
            image.save('./output_image.png')
            

    print('Finished validation.')
    return losses.avg



    

if __name__ == '__main__':
    
    writer = SummaryWriter('logs/0617')
    
    # 输入数据并处理
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

    trainset = BlackColorImages(root='../data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = BlackColorImages(root='../data/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    print("Data loaded.")
    
    # 初始化模型，训练参数
    model = ColorizationNet()
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model = model.cuda()
    criterion = criterion.cuda()
    
            
    # 测试模型
    test_only = False
    if test_only:
        model.load_state_dict(torch.load('./model_best.pth'))
        with torch.no_grad():
            validate(testloader, model, criterion, save_images=True)
            
    else:
        
        # 模型训练
        best_losses = float('inf')
        for epoch in range(100):
            train(trainloader, model, criterion, optimizer, epoch, writer)
            with torch.no_grad():
                losses = validate(testloader, model, criterion, save_images=True)
                writer.add_scalar('Loss/test', losses, epoch)
                # Save checkpoint and replace old best model if current model is better
                if losses < best_losses:
                    best_losses = losses
                    torch.save(model.state_dict(),'./model_best.pth')
        
        
        validate(testloader, model, criterion, save_images=True)