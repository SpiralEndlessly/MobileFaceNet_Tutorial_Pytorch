#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
Train Mobilefacenet

@author: AIRocker
"""
import os 
import sys
sys.path.append('..')
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from data_set.dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
from face_model import MobileFaceNet, Arcface
import time
from Evaluation import getFeature, evaluation_10_fold

def load_data(batch_size):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]

    root = 'data_set/CASIA/'
    file_list = 'data_set/CASIA/file_list_reduced.txt'
    dataset_train = CASIAWebFace(root, file_list, transform=transform)
    
    dataloader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    dataset_size = len(dataset_train)
    
    print('training and validation data loaded')
    
    return dataloader, dataset_size, dataset_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face_Detection_Training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training and evaluation')
    parser.add_argument('--epoch', type=int, default=12, help='number of epoches for training')
    parser.add_argument('--method', type=str, default='l2_distance', 
                            help='methold to evaluate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    dataloader, dataset_size, dataset = load_data(args.batch_size)
    model = MobileFaceNet(latent_size=320).to(device)
    print('MobileFaceNet face detection model loaded')
    margin = Arcface(embedding_size=128, classnum=int(dataset.class_nums),  s=32., m=0.5).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.01, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 8, 10], gamma=0.3) 
    start = time.time()
    ## save logging and weights
    train_logging_file = 'train_logging.txt'

    save_dir = 'checkpoints'
    if os.path.exists(save_dir):
        raise NameError('checkpoint dir exists!')
    os.makedirs(save_dir)

    total_iters = 0
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        model.train()     
        since = time.time()
        for det in dataloader: 
            img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()
            
            with torch.set_grad_enabled(True):
                raw_out = model(img)
                logits = torch.nn.functional.normalize(raw_out)
                output = margin(logits, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer_ft.step()
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy 
                    _, preds = torch.max(output.data, 1)
                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer_ft.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr)+'\n')
                    f.close()
                    
            # save model
            if total_iters % 3000 == 0:

                torch.save({
                    'iters': total_iters,
                    'net_state_dict': model.state_dict()},
                    os.path.join(save_dir, 'Iter_%06d_model.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin.state_dict()},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
        exp_lr_scheduler.step()
        
    time_elapsed = time.time() - start  
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))