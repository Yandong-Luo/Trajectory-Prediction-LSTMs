import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import time as t
import pandas as pd
from LSTM.utils import DataCenter, maskedMSE
import math
from tqdm import tqdm
from rich.progress import track
from torch.utils.data import DataLoader
from LSTM.model import TrajectoryNetwork
# from torch.utils.tensorboard import SummaryWriter  

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('learning_rate', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def train(lstm_config, tb_writer, dataset_name, resume = False):
    train_set = DataCenter(lstm_config, dataset_name, 'train')
    train_Dataloader = DataLoader(train_set,batch_size=lstm_config['batch_size'],shuffle=True,num_workers=8,collate_fn=train_set.collate_fn)

    valid_set = DataCenter(lstm_config, dataset_name, 'valid')
    valid_Dataloader = DataLoader(valid_set,batch_size=lstm_config['batch_size'],shuffle=True,num_workers=8,collate_fn=train_set.collate_fn)

    best_valid_loss = 10000
    best_lat_loss = 10000
    best_long_loss = 10000

    # initialize the network
    network = TrajectoryNetwork(lstm_config)

    if lstm_config['enable_cuda'] == True:
        network = network.cuda()

    ## Initialize optimizer
    # optimizer = torch.optim.Adam(network.parameters()) #lr = ...

    optimizer = torch.optim.AdamW(params=network.parameters(), 
                                  lr=lstm_config['learning_rate'], 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=lstm_config['learning_rate']*10, 
                                                    total_steps=len(train_Dataloader) * lstm_config['trainEpochs'], 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    
    start_epoch = 0
    if resume:
        checkpoint_path = os.path.join(lstm_config['saved_ckpt_path'], f'{dataset_name}_best_valid.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            network.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_valid_loss = checkpoint['loss']
            best_lat_loss = checkpoint['lat_loss']
            best_long_loss = checkpoint['longitude_loss']
            start_epoch = checkpoint['epoch'] + 1
            print(f"loaded previous checkpoint, start from epoch: {start_epoch}")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=lstm_config['learning_rate']*10, 
                                                    total_steps=len(train_Dataloader) * lstm_config['trainEpochs'], 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10,
                                                    last_epoch=start_epoch-1 if start_epoch != 0 else 0)
        else:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}', cannot resume training.")
        
    
    
    for epoch in range(start_epoch, lstm_config['trainEpochs']):
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_loss = 0
        avg_lon_loss = 0
        train_step = 0
        network = network.train()
        count = 0
        for i, data in enumerate(track(train_Dataloader, description=f"Training {dataset_name} Epoch {epoch+1}/{lstm_config['trainEpochs']}")):
            start_time = t.time()
            _, time, veh_id, velocity, acc, movement, history, future, neighbors, mask, output_masks = data

            if lstm_config['enable_cuda']:
                history = history.cuda()
                future = future.cuda()
                neighbors = neighbors.cuda()
                mask = mask.cuda()
                output_masks = output_masks.cuda()

            predict_result = network(history, neighbors, mask, True if dataset_name == 'peachtree' else True)
            loss, x_accuracy, y_accuracy = maskedMSE(predict_result, future, output_masks)

            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            # total_grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
            optimizer.step()
            scheduler.step()
            #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
            # Track average train loss and average train time:
            batch_time = t.time() - start_time
            avg_tr_loss += loss.item() # sum mse for 100 batches
            avg_tr_time += batch_time
            avg_lat_loss += x_accuracy.item()
            avg_lon_loss += y_accuracy.item()

            global_step = epoch * len(train_Dataloader) + train_step + 1

            if train_step % 10 == 0:
                loss_dict = {
                    'Avg train loss': avg_tr_loss / 10,
                    'Avg lateral loss': avg_lat_loss / 10,
                    'Avg longitude loss': avg_lon_loss / 10,
                    # 'Gradient norm': total_grad_norm
                }
                avg_tr_loss = 0 # clear the result every 10 batches
                avg_lat_loss = 0
                avg_lon_loss = 0
                avg_tr_time = 0
                save_summary(tb_writer, loss_dict, global_step, f'{dataset_name} train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])

            train_step += 1
            count += 1

        if valid_Dataloader is not None:
            network = network.eval()
            with torch.no_grad():
                avg_valid_loss = 0
                avg_valid_lat_loss = 0
                avg_valid_lon_loss = 0

                for i, data in enumerate(track(valid_Dataloader, description=f"Validation {dataset_name} Epoch {epoch+1}/{lstm_config['trainEpochs']}")):
                    _, time, veh_id, velocity, acc, movement, history, future, neighbors, mask, output_masks = data

                    if lstm_config['enable_cuda']:
                        history = history.cuda()
                        future = future.cuda()
                        neighbors = neighbors.cuda()
                        mask = mask.cuda()
                        output_masks = output_masks.cuda()

                    predict_valid_result = network(history, neighbors, mask, True if dataset_name == 'peachtree' else True)
                    loss, x_accuracy, y_accuracy = maskedMSE(predict_valid_result, future, output_masks)

                    # Backprop and update weights
                    optimizer.zero_grad()

                    avg_valid_loss += loss.item() # sum mse for 100 batches
                    avg_valid_lat_loss += x_accuracy.item()
                    avg_valid_lon_loss += y_accuracy.item()

                    # global_step = epoch * len(valid_Dataloader) + valid_step + 1


                    # if train_step % 10 == 0:
                loss_dict = {
                    'Avg valid loss': avg_valid_loss / len(valid_Dataloader),
                    'Avg valid lateral loss': avg_valid_lat_loss / len(valid_Dataloader),
                    'Avg valid longitude loss': avg_valid_lon_loss / len(valid_Dataloader),
                }
                save_summary(tb_writer, loss_dict, global_step, f'{dataset_name} valid')

                checkpoint = {
                    "net": network.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'loss': avg_valid_loss,
                    'lat_loss': avg_lat_loss,
                    'longitude_loss': avg_lon_loss
                }
                torch.save(checkpoint, os.path.join(lstm_config['saved_ckpt_path'], f'{dataset_name}_epoch_{epoch+1}.pth'))
                
                if best_valid_loss > avg_valid_loss / len(valid_Dataloader):
                    torch.save(checkpoint, os.path.join(lstm_config['saved_ckpt_path'], f'{dataset_name}_best_valid.pth'))