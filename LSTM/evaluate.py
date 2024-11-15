import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from LSTM.model import TrajectoryNetwork
from LSTM.utils import DataCenter, maskedMSETest, maskedMAETest

def evaluation(dataset_name, lstm_config, df=None, spec_file='pkl_file'):
    """
    evaluate the model based on MAE, MSE, RMSE. When df is not None, Generate a DataFrame of predicted trajectories.
    
    Args:
        model: The trained PyTorch model.
        dataset_name: The name of dataset: 'US101', 'I80', 'peachtree'
        lstm_config: Configuration dictionary.
        df: the dataframe from test pkl file
        
    Returns:
        predicted_df: A DataFrame with predicted Local_X and Local_Y for each vehicle at each frame.
    """
    model = TrajectoryNetwork(lstm_config)

    if lstm_config['enable_cuda'] == True:
        model = model.cuda()

    checkpoint_path = os.path.join(lstm_config['saved_ckpt_path'], f'{dataset_name}_best_valid.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['net'])
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}', cannot resume training.")

    # Set the model to evaluation mode
    model.eval()  

    test_set = DataCenter(lstm_config, dataset_name, 'test', spec_file)
    test_Dataloader = DataLoader(test_set, batch_size=lstm_config['batch_size'], shuffle=False, num_workers=8, collate_fn=test_set.collate_fn)

    if df is not None:
        df.insert(len(df.columns), 'Predicted_X', None)
        df.insert(len(df.columns), 'Predicted_Y', None)

    MSE_lossVal = 0
    MAE_lossVal = 0
    MSE_countVal = 0
    MAE_countVal = 0

    with torch.no_grad():
        for data in test_Dataloader:
            _, time, veh_id, velocity, acc, movement, history, future, neighbors, mask, output_masks = data

            if lstm_config['enable_cuda']:
                history = history.cuda()
                future = future.cuda()
                neighbors = neighbors.cuda()
                mask = mask.cuda()
                output_masks = output_masks.cuda()

            # Get predicted trajectory
            predicted_trajectory = model(history, neighbors, mask)

            MSE_loss, MSE_count = maskedMSETest(predicted_trajectory, future, output_masks)

            MAE_loss, MAE_count = maskedMAETest(predicted_trajectory, future, output_masks)

            try:
                MSE_lossVal += MSE_loss.detach()
            except:
                print("MSE_LOSS",MSE_loss)
            MAE_lossVal += MAE_loss.detach()

            MSE_countVal += MSE_count.detach()

            MAE_countVal += MAE_count.detach()

            if df is not None:
                # Convert predictions to CPU and numpy format for easy handling
                predicted_trajectory = predicted_trajectory.detach().cpu().numpy()

                # Loop through the batch to store predictions
                for i in range(predicted_trajectory.shape[0]):
                    vehicle_id = veh_id[i]
                    pred_x = predicted_trajectory[i, :, 0].tolist()  # X coordinates for 10 steps
                    pred_y = predicted_trajectory[i, :, 1].tolist()  # Y coordinates for 10 steps

                    row_index = df.index[(df['Vehicle_ID'] == vehicle_id) & (df['Global_Time'] == time[i])]

                    df.at[row_index[0], 'Predicted_X'] = pred_x
                    df.at[row_index[0], 'Predicted_Y'] = pred_y

        print(f"========================{dataset_name}========================")
        print ('Overall MSE is:', MSE_lossVal / MSE_countVal)
        print ('Overall RMSE is:', torch.sqrt(MSE_lossVal / MSE_countVal))
        print ('Overall MAE is:', MAE_lossVal / MAE_countVal)
        print("======================================================")
            
    return df