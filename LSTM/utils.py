import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from LSTM.model import TrajectoryNetwork

def write_pickle(results, file_path):
    """
    write the dataframe to pkl file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def read_pickle(file_path, filename='.pkl'):
    """
    reading the data from pkl file
    """
    assert os.path.splitext(file_path)[1] == filename
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def combine_US101_data(LSTM_config, prefix):
    """
    Combine all US101 data to .pkl file and rename the same vehicle ID in different time dataset
    """
    dataset_path = LSTM_config['dataset_path']
    US101_df_1 = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['US101']['time1_file']))
    US101_df_2 = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['US101']['time2_file']))
    US101_df_3 = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['US101']['time3_file']))

    # Convert Vehicle_ID columns to string type to allow renaming with suffixes
    US101_df_1['Vehicle_ID'] = US101_df_1['Vehicle_ID'].astype(str)
    US101_df_2['Vehicle_ID'] = US101_df_2['Vehicle_ID'].astype(str)
    US101_df_3['Vehicle_ID'] = US101_df_3['Vehicle_ID'].astype(str)

    # Extract unique Vehicle_IDs from each dataset
    vehicle_ids_1 = set(US101_df_1['Vehicle_ID'].unique())
    vehicle_ids_2 = set(US101_df_2['Vehicle_ID'].unique())
    vehicle_ids_3 = set(US101_df_3['Vehicle_ID'].unique())

    # Find common Vehicle_IDs
    common_vehicle_ids = vehicle_ids_1 & vehicle_ids_2 & vehicle_ids_3

    # Rename common Vehicle_IDs in df2 and df3 to ensure uniqueness
    for common_id in common_vehicle_ids:
        # Generate unique new IDs for each dataset
        US101_df_2.loc[US101_df_2['Vehicle_ID'] == common_id, 'Vehicle_ID'] = f"{common_id}_2"
        US101_df_3.loc[US101_df_3['Vehicle_ID'] == common_id, 'Vehicle_ID'] = f"{common_id}_3"

    df = pd.concat([US101_df_1, US101_df_2, US101_df_3], ignore_index=True)
    us101_path = os.path.join(dataset_path, f'us101_{prefix}_data.pkl')
    write_pickle(df, us101_path)

def combine_all_data(LSTM_config, prefix):
    """
    Combine all data to one .pkl file and rename the same vehicle ID in different datasets to ensure unique IDs.
    """
    dataset_path = LSTM_config['dataset_path']

    # Load data for each location
    US101_data = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['US101']['pkl_file']))
    I80_data = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['I80']['pkl_file']))
    peachtree_data = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][prefix]['peachtree']['pkl_file']))

    # Convert Vehicle_ID columns to string to allow renaming with suffixes
    US101_data['Vehicle_ID'] = US101_data['Vehicle_ID'].astype(str)
    I80_data['Vehicle_ID'] = I80_data['Vehicle_ID'].astype(str)
    peachtree_data['Vehicle_ID'] = peachtree_data['Vehicle_ID'].astype(str)

    # Extract unique Vehicle_IDs from each dataset
    vehicle_ids_US101 = set(US101_data['Vehicle_ID'].unique())
    vehicle_ids_I80 = set(I80_data['Vehicle_ID'].unique())
    vehicle_ids_peachtree = set(peachtree_data['Vehicle_ID'].unique())

    # Find common Vehicle_IDs across datasets and rename them
    common_vehicle_ids_US101_I80 = vehicle_ids_US101 & vehicle_ids_I80
    common_vehicle_ids_US101_peachtree = vehicle_ids_US101 & vehicle_ids_peachtree
    common_vehicle_ids_I80_peachtree = vehicle_ids_I80 & vehicle_ids_peachtree

    # Rename common Vehicle_IDs to ensure uniqueness
    for common_id in common_vehicle_ids_US101_I80:
        I80_data.loc[I80_data['Vehicle_ID'] == common_id, 'Vehicle_ID'] = f"{common_id}_I80"

    for common_id in common_vehicle_ids_US101_peachtree:
        peachtree_data.loc[peachtree_data['Vehicle_ID'] == common_id, 'Vehicle_ID'] = f"{common_id}_peachtree"

    for common_id in common_vehicle_ids_I80_peachtree:
        peachtree_data.loc[peachtree_data['Vehicle_ID'] == common_id, 'Vehicle_ID'] = f"{common_id}_peachtree"

    # Concatenate all datasets into one
    combined_df = pd.concat([US101_data, I80_data, peachtree_data], ignore_index=True)

    # Save the combined data as a .pkl file
    combined_path = os.path.join(dataset_path, f'all_{prefix}_data.pkl')
    write_pickle(combined_df, combined_path)

# def generate_predicted_df(dataset_name, lstm_config, df, spec_file='pkl_file'):
#     """
#     Generate a DataFrame of predicted trajectories.
    
#     Args:
#         model: The trained PyTorch model.
#         dataset_name: The name of dataset: 'US101', 'I80', 'peachtree'
#         lstm_config: Configuration dictionary.
#         df: the dataframe from test pkl file
        
#     Returns:
#         predicted_df: A DataFrame with predicted Local_X and Local_Y for each vehicle at each frame.
#     """
#     model = TrajectoryNetwork(lstm_config)

#     if lstm_config['enable_cuda'] == True:
#         model = model.cuda()

#     checkpoint_path = os.path.join(lstm_config['saved_ckpt_path'], f'{dataset_name}_best_valid.pth')

#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, weights_only=True)
#         model.load_state_dict(checkpoint['net'])
#     else:
#         raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}', cannot resume training.")

#     # Set the model to evaluation mode
#     model.eval()  

#     test_set = DataCenter(lstm_config, dataset_name, 'test', spec_file)
#     test_Dataloader = DataLoader(test_set, batch_size=lstm_config['batch_size'], shuffle=False, num_workers=8, collate_fn=test_set.collate_fn)

#     df.insert(len(df.columns), 'Predicted_X', None)
#     df.insert(len(df.columns), 'Predicted_Y', None)


#     with torch.no_grad():
#         for data in test_Dataloader:
#             _, time, veh_id, velocity, acc, movement, history, _, neighbors, mask, output_masks = data

#             if lstm_config['enable_cuda']:
#                 history = history.cuda()
#                 neighbors = neighbors.cuda()
#                 mask = mask.cuda()

#             # Get predicted trajectory
#             predicted_trajectory = model(history, neighbors, mask)

#             # Convert predictions to CPU and numpy format for easy handling
#             predicted_trajectory = predicted_trajectory.detach().cpu().numpy()

#             # Loop through the batch to store predictions
#             for i in range(predicted_trajectory.shape[0]):
#                 vehicle_id = veh_id[i]
#                 pred_x = predicted_trajectory[i, :, 0].tolist()  # X coordinates for 10 steps
#                 pred_y = predicted_trajectory[i, :, 1].tolist()  # Y coordinates for 10 steps

                
#                 # Match `Vehicle_ID` and the latest `Frame_ID` in df for insertion
#                 # mask = (df['Vehicle_ID'] == vehicle_id) & (df['Global_Time'] == time[i])
#                 row_index = df.index[(df['Vehicle_ID'] == vehicle_id) & (df['Global_Time'] == time[i])]
                
#                 # Store the predicted positions for this vehicle and frame
#                 # df.loc[row_index[0], 'Predicted_X'] = pred_x
#                 # df.loc[row_index[0], 'Predicted_Y'] = pred_y

#                 df.at[row_index[0], 'Predicted_X'] = pred_x
#                 df.at[row_index[0], 'Predicted_Y'] = pred_y
            
#     return df


def load_us101_train_data(LSTM_config):
    dataset_path = LSTM_config['dataset_path']
    US101_df = read_pickle(os.path.join(dataset_path, LSTM_config['dataset']['train']['US101']['all_file']))
    return US101_df

class DataCenter(Dataset):
    def __init__(self, LSTM_config, dataset_name, split_name, spec_file='pkl_file'):
        self.LSTM_config = LSTM_config

        self.dataset_name = dataset_name

        dataset_path = LSTM_config['dataset_path']

        self.data = read_pickle(os.path.join(dataset_path, LSTM_config['dataset'][split_name][dataset_name][spec_file]))

        self.data.reset_index(drop=True, inplace=True)

        self.predict_step = LSTM_config['predict_step']
        self.history_step = LSTM_config['history_step']
        self.encode_size = LSTM_config['encode_size']

        grid_size_str = LSTM_config['grid_size']  # e.g., "(13, 3)"
        grid_size_tuple = tuple(map(int, grid_size_str.strip("()").split(",")))
        self.grid_size = np.array(grid_size_tuple)

        # when you change the output of getHistory() or getFuture(), please check this value is it correct or not.
        self.len_status = 7     # 'Global_Time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Theta', 'Movement'

        self.vehicle_traj_dict = {}
        for vehicle_id, group in self.data.groupby('Vehicle_ID'):
            self.vehicle_traj_dict[vehicle_id] = group.sort_values('Global_Time')

        # print(self.vehicle_traj_dict.keys())

    def __getitem__(self, idx):
        veh_id = self.data.loc[idx, 'Vehicle_ID']
        time = self.data.loc[idx, 'Global_Time']
        speed = self.data.loc[idx, 'v_Vel']
        acc = self.data.loc[idx, 'v_Acc']
        movement = self.data.loc[idx, 'Movement']
        grid = self.data.loc[idx, 'Grid_Neighbors']
        yaw = self.data.loc[idx, 'Theta']

        history_states = self.getHistory(veh_id, veh_id, time)
        future_states = self.getFuture(veh_id, veh_id, time)

        neighbors = []
        # if self.dataset_name != 'peachtree':
        for neighbor_id in grid:
            if neighbor_id != None:
                neighbor_history = self.getHistory(neighbor_id, veh_id, time)
                neighbors.append(neighbor_history)

        return time, veh_id, speed, acc, movement, yaw, history_states, future_states, neighbors
    
    def __len__(self):
        return len(self.data)

    def getHistory(self, veh_id, ref_id, time):
        if veh_id not in self.vehicle_traj_dict.keys():
            return np.empty([0, self.len_status])
        veh_traj = self.vehicle_traj_dict[veh_id]

        ref_traj = self.vehicle_traj_dict[ref_id]
        refData = ref_traj[ref_traj['Global_Time'] == time]
        # print(refData)
        # print(time, refData['Local_X'], refData['Local_Y'], refData['v_Vel'], refData['v_Acc'])
        refState = np.array([refData.iloc[0]['Local_X'], refData.iloc[0]['Local_Y'], refData.iloc[0]['v_Vel'], refData.iloc[0]['v_Acc'], refData.iloc[0]['Theta'], refData.iloc[0]['Global_Time']])
        # print("refState", refState)
        # refMovement = refData['Movement']
        
        # get history state: position, time, speed, acc
        # end_idx = veh_traj.index[veh_traj['Global_Time'] == time][0]
        # start_idx = max(0, end_idx - self.history_step)
        # print("start_idx", start_idx, "end_idx", end_idx, "len", len(veh_traj))
        # past_traj = veh_traj.iloc[start_idx:end_idx + 1]
        past_traj = veh_traj[(veh_traj['Global_Time'] >= time - self.history_step * 100) & 
                     (veh_traj['Global_Time'] <= time)]

        # print("past_traj", past_traj)
        history_states = past_traj[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Theta', 'Global_Time']].to_numpy()
        history_movement = past_traj[['Movement']].to_numpy()
        
        # the relative status
        relative_states = history_states - refState

        history_relative_states = np.hstack((relative_states, history_movement))

        # print("history states shape", history_relative_states.shape)
        # print("history states", history_relative_states)

        return history_relative_states
    
    def getFuture(self, veh_id, ref_id, time):
        veh_traj = self.vehicle_traj_dict[veh_id]

        ref_traj = self.vehicle_traj_dict[ref_id]
        refData = ref_traj[ref_traj['Global_Time'] == time]
        refState = np.array([refData.iloc[0]['Local_X'], refData.iloc[0]['Local_Y'], refData.iloc[0]['v_Vel'], refData.iloc[0]['v_Acc'], refData.iloc[0]['Theta'], refData.iloc[0]['Global_Time']])

        # get future state: position, time, speed, acc
        # start_idx = veh_traj.index[veh_traj['Global_Time'] == time][0]
        # end_idx = min(len(veh_traj), start_idx + self.predict_step)
        # future_traj = veh_traj.iloc[start_idx:end_idx]
        future_traj = veh_traj[(veh_traj['Global_Time'] <= time + self.predict_step * 100) & 
                     (veh_traj['Global_Time'] > time)]
        future_states = future_traj[[ 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Theta', 'Global_Time']].to_numpy()
        future_movement = future_traj[['Movement']].to_numpy()
        
        # the relative status
        relative_states = future_states - refState

        future_relative_states = np.hstack((relative_states, future_movement))

        return future_relative_states
    
    def collate_fn(self, samples):
        """Collate function for dataloader"""
        neighbor_batch_size = sum([sum([len(neighbor) != 0 for neighbor in sample[-1]]) for sample in samples])
        
        if neighbor_batch_size == 0:
            neighbor_batch_size = 39

        # just only record valid neighbor to save the space
        neighbor_batch = torch.zeros(neighbor_batch_size, self.history_step + 1, self.len_status)

        # mark which cells in the grid have valid neighbor data
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.encode_size)
        mask_batch = mask_batch.byte()

        # output mask
        output_masks = torch.zeros(len(samples), self.predict_step, 2)

        # velocity_batch = torch.zeros(len(samples), 1)
        # acc_batch = torch.zeros(len(samples), 1)
        # movement_batch = torch.zeros(len(samples), 1)
        velocity_batch = []
        acc_batch = []
        movement_batch = []
        veh_id_batch = []
        time_batch = []
        yaw_batch = []
        history_batch = torch.zeros(len(samples), self.history_step + 1, self.len_status)
        future_batch = torch.zeros(len(samples), self.predict_step, self.len_status)

        count = 0

        for i, (time, veh_id, speed, acc, movement, yaw, history_states, future_states, neighbors) in enumerate(samples):
            time_batch.append(time)
            veh_id_batch.append(veh_id)
            velocity_batch.append(speed)
            acc_batch.append(acc)
            yaw_batch.append(yaw)
            movement_batch.append(movement)
            history_batch[ i, :len(history_states), :] = torch.from_numpy(history_states[:,:])
            future_batch[i, :len(future_states), :] = torch.from_numpy(future_states[:, :])
            output_masks[i, :len(future_states), :] = 1

            for neighbor_id, neighbor in enumerate(neighbors):
                if len(neighbor) != 0:
                    neighbor_batch[count, :len(neighbor), :] = torch.from_numpy(neighbor[:,:])
                    mask_batch[i, neighbor_id // self.grid_size[0], neighbor_id % self.grid_size[0], :] = torch.ones(self.encode_size).byte()
                    count += 1

        return len(samples), time_batch, veh_id_batch, velocity_batch, acc_batch, movement_batch, history_batch, future_batch, neighbor_batch, mask_batch, output_masks
    


## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)

    # Extract x and y predictions
    predict_X = y_pred[:, :, 0]
    predict_Y = y_pred[:, :, 1]

    # Extract ground truth x and y
    Truth_X = y_gt[:, :, 0]
    Truth_Y = y_gt[:, :, 1]

    # Calculate squared error for x and y
    error_x = torch.pow(Truth_X - predict_X, 2)
    error_y = torch.pow(Truth_Y - predict_Y, 2)

    # Apply the mask to each error
    masked_error_x = error_x * mask[:, :, 0]
    masked_error_y = error_y * mask[:, :, 1]

    # Calculate the mean squared error for x and y, considering only valid values
    x_accuracy = torch.sum(masked_error_x) / torch.sum(mask[:, :, 0])
    y_accuracy = torch.sum(masked_error_y) / torch.sum(mask[:, :, 1])

    # Calculate the overall loss if needed
    lossVal = (x_accuracy + y_accuracy) / 2

    return lossVal, x_accuracy, y_accuracy

    # out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    # acc[:,:,0] = out 
    # acc[:,:,1] = out
    # acc = acc*mask
    # lossVal = torch.sum(acc)/torch.sum(mask) # although both uses out, the average will be correct, 2*out/2
    # return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)

    acc = out * mask[:,:,0]
    lossVal = torch.sum(acc)
    counts = torch.sum(mask[:,:,0])
    return lossVal, counts

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMAETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)), 0.5)
    acc = out * mask[:, :, 0]
    lossVal = torch.sum(acc)
    counts = torch.sum(mask[:,:,0])
    return lossVal, counts