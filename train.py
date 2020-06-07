# -*- coding:utf-8 -*-
#@Time  : 2020/6/6 15:16
#@Author: DongBinzhi
#@File  : train.py

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from net import Net

dir_train_data = 'dataset/train_data.csv'
dir_test_data = 'dataset/test_data.csv'
dir_checkpoint = 'checkpoints/'

def eval_net(net, loader, device, batch_size):
    """Evaluation model on test dataset"""
    net.eval()
    n_val = len(loader)  # the number of batch
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    val_acc = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            data, true_labels = batch['data'], batch['label']
            data = data.to(device=device, dtype=torch.float32)
            true_labels = true_labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                labels_pred = net(data)

            loss = criterion(labels_pred, true_labels.long())
            val_loss += loss.data.item()*true_labels.size(0)
            _, pred = torch.max(labels_pred, 1)
            num_correct = (true_labels == pred).sum()
            val_acc += num_correct.item()
            pbar.update()

    net.train()
    return  val_loss / (n_val*batch_size), val_acc / (n_val*batch_size)



def train_net(net,
              device,
              epochs=5,
              batch_size=512,
              lr=0.0001,
              save_cp=True):

    # create the dataloader
    train_dataset = BasicDataset(data_path=dir_train_data, train_flag=True)
    val_dataset = BasicDataset(data_path = dir_test_data, train_flag=False)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    n_train = len(train_loader)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit=' record') as pbar:
            for batch in train_loader:
                data = batch['data']
                true_labels = batch['label']


                data = data.to(device=device, dtype=torch.float32)
                true_labels = true_labels.to(device=device, dtype=torch.float32)

                labels_pred = net(data)
                
                loss = criterion(labels_pred, true_labels.long())
                
                epoch_loss += loss.item()
                
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.001)
                optimizer.step()

                pbar.update(data.shape[0])
                global_step += 1
                if global_step % (len(train_dataset) // ( 5*batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    

                    val_loss, val_acc = eval_net(net, val_loader, device,batch_size)
                    
                    scheduler.step(val_acc)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation accuracy: {}'.format(val_acc))
                    writer.add_scalar('Accuracy/test', val_acc, global_step)


                    logging.info('Validation loss: {}'.format(val_loss))
                    writer.add_scalar('Loss/test', val_loss, global_step)
                    
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()







def get_args():
    parser = argparse.ArgumentParser(description='Train the DNN on KDD Cup 1999.  Note: the default parameters are not the best!!!',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=512,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Net(input_dim=28, hidden_1=256, hidden_2=512, hidden_3=256, out_dim=5)
    print(net)
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

