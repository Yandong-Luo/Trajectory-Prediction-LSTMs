import numpy as np
import pickle

import pandas as pd
import multiprocessing
from time import sleep
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import os
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# add current path to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MultiThread_FeatureProcess:
    def __init__(self, df_dict, config) -> None:
        self.df_dict = df_dict
        self.config = config

    def run(self):
        # Use the number of CPU cores available on the system
        n_workers = multiprocessing.cpu_count()

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            futures = []
            # Share state between processes
            with multiprocessing.Manager() as manager:
                _progress = manager.dict()
                overall_progress_task = progress.add_task("[green]All jobs progress:")

                # Use ProcessPoolExecutor for multiprocessing
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    for name, df in self.df_dict.items():
                        if name in ['peachtree', 'I80']:
                            continue
                        # Create a new task for each dataset
                        task_id = progress.add_task(f"Processing {name}", visible=False)
                        futures.append(executor.submit(
                            self.multi_thread_feature_process, name, df, _progress, task_id
                        ))

                    # Monitor progress
                    while (n_finished := sum([future.done() for future in futures])) < len(futures):
                        
                        total_progress = 0
                        for task_id, update_data in _progress.items():
                            latest = update_data["progress"]
                            total = update_data["total"]
                            total_progress += latest
                            # Update progress for each task
                            progress.update(task_id, completed=latest, total=total, visible=latest < total,)
                        progress.update(
                            overall_progress_task, completed=total_progress/3, total=100
                        )

                    # Check for errors
                    for future in futures:
                        try:
                            future.result()  # This will raise the exception if the task failed
                        except Exception as e:
                            logging.error(f"Error in process: {e}")

    def feature_process(self, df, vehicle_traj_dict, vehicle_time_dict, name):
        movement_batch = []
        neighbors_batch = []
        theta_batch = []
        for idx, data in df.iterrows():
            time = data['Global_Time']
            vehicle_id = data['Vehicle_ID']
            veh_traj = vehicle_traj_dict[vehicle_id]
            veh_time = vehicle_time_dict[time]
            ego_lane_id = data['Lane_ID']
            ego_intersection_id = data['Int_ID']
            ego_local_x = data['Local_X']
            ego_local_y = data['Local_Y']
            ego_section_id = data['Section_ID']
            ego_direction = data['Direction']
            # Find the index for the current time frame
            # df_idx = df[df['Global_Time'] == time & df['Vehicle_ID'] == vehicle_id].index[0]
            # logging.info(f"Start getting current idx of vehicle {vehicle_id}")
            current_idx = veh_traj[veh_traj['Global_Time'] == time].index.tolist()[0]

            # Update movement for US101
            if name != 'I80' and name != 'peachtree':
                try:
                    movement = self.generate_movement(veh_traj, current_idx)
                except Exception as e:
                    # print("current_idx",current_idx)
                    # print(veh_traj[veh_traj['Global_Time'] == time].index.tolist())
                    logging.error(f"Error processing when getting the movement of vehicle {vehicle_id}: {e}")

            # add neighbor feature for each dataset
            try:
                neighbors = self.generate_neighbors(df, name, veh_time, ego_lane_id, ego_intersection_id, ego_local_x, ego_local_y, current_idx, ego_section_id, ego_direction)
            except Exception as e:
                logging.error(f"Error processing when getting the neighbors of vehicle {vehicle_id}: {e}")
            
            try:
                # print("try to calculate theta")
                theta = self.calculate_theta(ego_local_x, ego_local_y, veh_traj, time)
            except Exception as e:
                logging.error(f"Error processing when calculate the theta of vehicle {theta}: {e}")
    
            if name != 'I80' and name != 'peachtree':
                movement_batch.append(movement)
    
            neighbors_batch.append(neighbors)
            theta_batch.append(theta)
    
        return movement_batch, neighbors_batch, theta_batch
    
    def multi_thread_feature_process(self, name, df, progress, task_id):
        vehicle_traj_dict = {}
        vehicle_time_dict = {}
        min_required_frames = 40
        # Group data by Vehicle_ID
        for vehicle_id, group in df.groupby('Vehicle_ID'):
            # Ensure that the vehicle has at least 3s of trajectory
            if len(group) >= min_required_frames:
                # Store the entire trajectory of each vehicle as a DataFrame
                vehicle_traj_dict[vehicle_id] = group.sort_values('Global_Time')
        
        valid_vehicle_ids = set(vehicle_traj_dict.keys())  # Only keep vehicles with 3s of trajectory
        df = df[df['Vehicle_ID'].isin(valid_vehicle_ids)].copy()  # Filter df to include only valid vehicles

        # Group data by Global_Time (frames)
        for time, group in df.groupby('Global_Time'):
            # Store all vehicles present at a given time frame as a DataFrame
            vehicle_time_dict[time] = group.sort_values('Vehicle_ID')
        
            
        # split the df for multi-thread feature process
        batch_size = min(self.config['feature_processing']['batch_size'], len(df))
        df_batch_list = []
        idx = 0
        while idx < len(df):
            df_batch_list.append(df.iloc[idx:idx + batch_size, :])
            idx += batch_size
        # logging.info(f"Completed spliting data processing {name}")
        # list for multi thread result
        movement_results = []
        neighbors_results = []
        theta_results = []
        with ThreadPoolExecutor(max_workers= os.cpu_count()) as thread_executor:
            futures = []
            for idx, df_batch in enumerate(df_batch_list):
                # thread_id = progress.add_task(f"Processing {name} Thread", visible=False)
                # logging.info(f"Start adding thread processing {name}")
                futures.append(thread_executor.submit(self.feature_process, df_batch, vehicle_traj_dict, vehicle_time_dict, name))
                progress[task_id] = {"progress": (idx + 1) * 100 / len(df_batch_list), "total": 100}
                # logging.info(f"Completed adding thread processing {name}")

            # combine all result from thread
            for future in futures:
                movement_batch, neighbors_batch, theta_batch = future.result()
                movement_results += movement_batch
                neighbors_results += neighbors_batch
                theta_results += theta_batch
        
        if len(movement_results) != len(df) or len(neighbors_results) != len(df) or len(theta_results) != len(df):
        # if len(neighbors_results) != len(df):
            print("length is error")
        if len(movement_results) != len(df):
            print("movement length error")
        elif len(neighbors_results) != len(df):
            print("neighbor length")
        elif len(theta_results) != len(df):
            print("theta length error")
        # merge the result to df
        if name != 'I80' and name != 'peachtree':
            try:
                df['Movement'] = movement_results
            except Exception as e:
                logging.error(f"Error processing when update the movement: {e}")

        try:
            df['Theta'] = theta_results
        except Exception as e:
            print("Theta len:",len(theta_results))
            print("df len:",len(df))
            logging.error(f"Error processing when update the theta: {e}")

        try:
            df['Grid_Neighbors'] = neighbors_results
        except Exception as e:
            print("all list?", all(isinstance(i, list) for i in neighbors_results))
            print("neighbor len:",len(neighbors_results))
            print("df len:",len(df))
            logging.error(f"Error processing when update the neighbor: {e}")
        
        try:
            self.df_dict[name] = df
        except Exception as e:
            logging.error(f"Error processing when update the whole df: {e}")

        if self.config['output']['save_data']:
            labels = ["train", "valid", "test"]
            if self.config['output']['save_data_for_model']:
                # normalization
                scalar_local_x = np.max(df['Local_X']) - np.min(df['Local_X'])
                scalar_local_y = np.max(df['Local_Y']) - np.min(df['Local_Y'])
                scalar_vel = np.max(df['v_Vel']) - np.min(df['v_Vel'])
                scalar_acc = np.max(df['v_Acc']) - np.min(df['v_Acc'])

                df.loc[:, 'Local_X'] = (df['Local_X'] - np.min(df['Local_X'])) / scalar_local_x
                df.loc[:, 'Local_Y'] = (df['Local_Y'] - np.min(df['Local_Y'])) / scalar_local_y
                df.loc[:, 'v_Vel'] = (df['v_Vel'] - np.min(df['v_Vel'])) / scalar_vel
                df.loc[:, 'v_Acc'] = (df['v_Acc'] - np.min(df['v_Acc'])) / scalar_acc

                df = df.drop(['v_length','v_Width','Direction','Location', 'Section_ID','Int_ID','Lane_ID', 'v_Class','Frame_ID'], axis=1)
            df_list = self.train_valid_test_split(df)
            for label, df in zip(labels, df_list):
                train_pkl_path = os.path.join(self.config['output']['pickle_output_folder'], f'{name}_{label}_data.pkl')
                self.write_pickle(df, train_pkl_path)
            logging.info(f"Completed writing data to pkl file {name}")

    def train_valid_test_split(self, df):
        # 70% for training
        flag_1 = round(0.7 * max(df['Vehicle_ID']))
        flag_2 = round(0.9 * max(df['Vehicle_ID']))

        train_df = df[df['Vehicle_ID'] <= flag_1]
        valid_df = df[(df['Vehicle_ID'] > flag_1) & (df['Vehicle_ID'] <= flag_2)]
        test_df = df[df['Vehicle_ID'] > flag_2]

        return [train_df, valid_df, test_df]
    
    def generate_movement(self, veh_traj, current_idx):
        """Due to the dataset of US101 stills lack of the movement,
        this function updates movement for US101
        """
        # lateral 
        # Each frame is 0.1s apart. we use current_idx + 40 means we consider the lateral position next 4s and past 4s.
        upper_boundary = min(current_idx + 40, veh_traj.index[-1])
        lower_boundary = max(veh_traj.index[0], current_idx - 40)
        
        movement = 1
        if veh_traj.loc[upper_boundary, 'Lane_ID'] > veh_traj.loc[current_idx, 'Lane_ID'] or \
            veh_traj.loc[current_idx, 'Lane_ID'] > veh_traj.loc[lower_boundary, 'Lane_ID']:
            movement = 3 # turning right
        elif veh_traj.loc[upper_boundary, 'Lane_ID'] < veh_traj.loc[current_idx, 'Lane_ID'] or \
            veh_traj.loc[current_idx, 'Lane_ID'] < veh_traj.loc[lower_boundary, 'Lane_ID']:
            movement = 2 # turning left
        else:
            movement = 1 # lane keeping
        
        return movement

    def add_neighbors(self, df, frame, ego_local_x, ego_local_y, current_idx, base_index, neighbors, intersection = False):
        """
        Get the neighbor based on left, right, and current lane
        Input:
            df: the data frame of current dataset
            frame: the data frame of left or right lane 
            ego_local_y: the Longitudinal distance
            current_idx: the row idx of ego in df
        """
        if intersection == False:
            if frame is not None and not frame.empty:  # Check if the frame contains any data
                for _, vehicle in frame.iterrows():
                    y_diff = vehicle['Local_Y'] - ego_local_y  # Vertical difference with ego vehicle
                    if abs(y_diff) < 90:  # Only consider vehicles within ±90 units
                        grid_index = int(base_index + round((y_diff + 90) / 15))
                        # Store the neighbor vehicle ID in the grid position
                        # neighbors[grid_index] = vehicle['Vehicle_ID']
                        # Ensure grid_index is within bounds of the neighbors list
                        if 0 <= grid_index < len(neighbors):
                            neighbors[grid_index] = vehicle['Vehicle_ID']
                        else:
                            print("base_index",base_index)
                            logging.error(f"Error: grid_index {grid_index} out of range for vehicle {vehicle['Vehicle_ID']}")

        else:
            if frame is not None and not frame.empty:
                for _, vehicle in frame.iterrows():
                    y_diff = vehicle['Local_Y'] - ego_local_y
                    x_diff = vehicle['Local_X'] - ego_local_x

                    if abs(x_diff) < 5 and abs(y_diff) < 45:
                        base_index = 14
                    elif x_diff < -5 and x_diff > -13 and abs(y_diff) < 45:
                        base_index = 1
                    elif x_diff > 5 and x_diff < -13 and abs(y_diff) < 45:
                        base_index = 27
                    else:
                        continue
                    
                    grid_index = base_index + round((y_diff + 90) / 15)
                    neighbors[grid_index] = vehicle['Vehicle_ID']
        
        return neighbors

    def generate_neighbors(self, df, name, veh_time, ego_lane_id, ego_intersection_id, ego_local_x, ego_local_y, current_idx, ego_section_id, ego_direction):
        """
        Feature Engineering 1: generate Neighbors
        Input:
            df: the data frame of current dataset
            veh_time: Dataframe sorted by time
            ego_lane_id: current lane id of ego
            ego_intersection_id: if ego stay at intersection, ego_intersection_id is the id of this intersection
            ego_local_x: lateral distance
            ego_local_y: the Longitudinal distance
            current_idx: the row idx of ego in df
        Output:
            df
        """
        ego_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
        farthest_left_lane = 1

        max_grid_size = 39
        neighbors = [ None for _ in range(max_grid_size)]

        if name == 'I80':
            farthest_right_lane = 6
        elif name == 'peachtree':
            farthest_right_lane = 3
        else:
            farthest_right_lane = 5
        
        if ego_section_id == 0:  # intersection
            neighbor_frame = veh_time[(veh_time['Lane_ID'] == 0) & (veh_time['Int_ID'] == ego_intersection_id) & (veh_time['Direction'] == ego_direction)]
            neighbors = self.add_neighbors(df, neighbor_frame, ego_local_x, ego_local_y, current_idx, 0, neighbors, intersection = True)
        else:
            left_frame = None
            right_frame = None

            if ego_lane_id > farthest_left_lane and ego_lane_id < farthest_right_lane:
                left_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id - 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                right_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id + 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
            elif ego_lane_id == 1:
                left_frame = veh_time[(veh_time['Lane_ID'] == 12) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                if left_frame.empty:
                    left_frame = veh_time[(veh_time['Lane_ID'] == 11) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                right_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id + 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
            elif ego_lane_id == 6:
                left_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id - 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                right_frame = None
            elif ego_lane_id == 11:
                right_frame = veh_time[(veh_time['Lane_ID'] == 12) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                if right_frame.empty:
                    right_frame = veh_time[(veh_time['Lane_ID'] == 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                left_frame = None
            elif ego_lane_id == 12:
                left_frame = veh_time[(veh_time['Lane_ID'] == ego_lane_id - 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
                right_frame = veh_time[(veh_time['Lane_ID'] == 1) & (veh_time['Section_ID'] == ego_section_id) & (veh_time['Direction'] == ego_direction)]
            
            neighbors = self.add_neighbors(df, left_frame, ego_local_x, ego_local_y, current_idx, 0, neighbors, intersection = False)
            neighbors = self.add_neighbors(df, ego_frame, ego_local_x, ego_local_y, current_idx, 13, neighbors, intersection = False)
            neighbors = self.add_neighbors(df, right_frame, ego_local_x, ego_local_y, current_idx, 26, neighbors, intersection = False)
        return neighbors
    
    def calculate_theta(self, local_x, local_y, veh_traj, time):
        """Calculates the heading theta using local coordinates."""
        # if veh_traj[veh_traj['Global_Time'] == time - 100]
        next_data = veh_traj[veh_traj['Global_Time'] == time +100 ]
        if not next_data.empty:
            # print(current_idx)
            # prev_data = veh_traj.loc[current_idx - 1]
            next_x = next_data.iloc[0]['Local_X']
            next_y = next_data.iloc[0]['Local_Y']

            delta_x = next_x - local_x
            delta_y = next_y - local_y
            theta = np.arctan2(delta_y, delta_x)  # calculate theta in radians
            theta_degree = np.degrees(theta)  # convert to degrees
        else:
            theta_degree = 0
        return theta_degree
    
    def write_pickle(self, results, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
