import numpy as np
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import yaml
from matplotlib.animation import FuncAnimation
from multithread_featureprocess import MultiThread_FeatureProcess
from LSTM.train import train
from LSTM.utils import combine_US101_data, combine_all_data
from LSTM.evaluate import evaluation
from torch.utils.tensorboard import SummaryWriter  
import sys
import os

# add current path to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

def load_configuration(yaml_path):
	""" loading .yaml file """
	with open(yaml_path, 'r') as file:
		config = yaml.safe_load(file)
	return config

def load_all_csv_path(config):
	"""
	loading all csv_path based on .yaml file
	Args:
		config: the object of yaml file
	Return:
		all_csv_path: the list of all csv path
		scenarios: the list of scenarios name
	"""

	dataset_path = config['dataset_path']
	
	# the dataset collected from Peachtree street, GA
	peachtree_folder = config['datasets']['Peachtree']['folder']
	peachtree_csv = config['datasets']['Peachtree']['csv_file']
	peachtree_csv_path = os.path.join(dataset_path, peachtree_folder, peachtree_csv)

	# the dataset collected from US101
	US101_folder = config['datasets']['US101']['folder']
	US101_time1_csv = config['datasets']['US101']['csv_files']['time1']
	US101_time2_csv = config['datasets']['US101']['csv_files']['time2']
	US101_time3_csv = config['datasets']['US101']['csv_files']['time3']
	US101_time1_csv_path = os.path.join(dataset_path, US101_folder, US101_time1_csv)
	US101_time2_csv_path = os.path.join(dataset_path, US101_folder, US101_time2_csv)
	US101_time3_csv_path = os.path.join(dataset_path, US101_folder, US101_time3_csv)

	# the dataset collected from I80
	I80_folder = config['datasets']['I80']['folder']
	I80_csv = config['datasets']['I80']['csv_file']
	I80_csv_path = os.path.join(dataset_path, I80_folder, I80_csv)

	all_csv_path = [US101_time1_csv_path, US101_time2_csv_path, US101_time3_csv_path, I80_csv_path, peachtree_csv_path]
	scenarios = ["us101_time1","us101_time2","us101_time3","I80","peachtree"]

	return all_csv_path, scenarios



def BatchData_Load_clearning(scenarios, all_csv_path):
	"""
	bach process dataset: cleaning, 
	"""
	df_dict = {}

	for name, csv_path in zip(scenarios, all_csv_path):
		# loading data
		df = pd.read_csv(csv_path)

		# drop the information we don't need. Due to there are some difference between different dataset, we need to adjust some data
		df = df.drop(['Global_X','Global_Y','Total_Frames','Following'], axis=1)
		if name == 'I80' or name == 'peachtree':
			
			df = df.drop(['O_Zone','D_Zone','Preceding'], axis=1)
			# df.insert(14, 'Action', 1)       # new feature: 1: maintaining speed, 2: slow down, 3:accelerate; still need to update
		else:
			# add Int_ID
			# add section id
			# add movement
			# add location
			df = df.drop('Preceeding', axis=1)
			df = df.rename(columns={"v_Length": "v_length"})
			# Due to US101 is a stright highway, there are no any intersection,
			# manually adding 'Int_ID' and 'Section_ID' to make sure all dataset consistent
			df.insert(11, 'Int_ID', 0)         # 0 Value of “0” means that the vehicle was not in the immediate vicinity of an intersection 
			df.insert(12, 'Section_ID', 101)   # 101 is a random value for US101 to mark it as "no Intersection". "0" means vehicle driving at intersection
			df.insert(13, 'Movement', 1)       # initialize the movement, they will be updated
			df.insert(14, 'Direction', 1)      # all vehicle in US101 use the same direction
			# df.insert(14, 'Action', 1)       # new feature: 1: maintaining speed, 2: slow down, 3:accelerate; still need to update
			df.insert(15, 'Theta', 0)          # initialize the theta as 0
			df.insert(df.shape[1], 'Location', name)

		# Initialize a new column for grid neighbors
		max_grid_size = 39  # Grid has a total of 33 slots (3x11 per lane)
		# empty_dict_template = {i: None for i in range(max_grid_size)}
		empty_neighbors = [ None for i in range(max_grid_size)]
		df.insert(15, 'Grid_Neighbors', [empty_neighbors.copy() for _ in range(len(df))])
		
		# Clear data containing NaN
		# df = df.dropna()
		# Clear different vehicle tracks of the same vehicle at the same time
		df = df.drop_duplicates(subset=['Vehicle_ID', 'Global_Time'])
		if name == 'I80':
			# only keep the land_id from 1 to 6,
			# 11 and 12 for Left turn lane
			# 0 for intersection
			df.query('6 >= Lane_ID >= 1 | Lane_ID == 0 | Lane_ID == 11 | Lane_ID == 12', inplace=True)

		elif name == 'peachtree':
			df.query('2 >= Lane_ID >= 1 | Lane_ID == 0 | Lane_ID == 11 | Lane_ID == 12', inplace=True)
		else:
			df.query('8 >= Lane_ID >= 1', inplace=True)

		df_dict[name] = df
	
	return df_dict

def write_pickle(results, file_path):
	with open(file_path, 'wb') as f:
		pickle.dump(results, f)

def read_pickle(file_path, filename='.pkl'):
	assert os.path.splitext(file_path)[1] == filename
	with open(file_path, 'rb') as f:
		data = pickle.load(f)
	return data


def plot_highway_frame(frame, ax, df, config, frames, ani, fig):
	ax.clear()

	# loading parameter from yaml file
	x_limit = config['visualization']['highway']['x_limit']
	y_limit = config['visualization']['highway']['y_limit']
	section_limits = config['visualization']['highway']['section_limits']

	frame_data = df[(df['Frame_ID'] == frame) &
					(df['Local_Y'] >= section_limits[0]) &
					(df['Local_Y'] <= section_limits[1])]

	# Avoid running for a long time, so end it at 300 frames
	if frame_data.empty:
		return

	# Get needed fields
	lateral_pos = frame_data['Local_X'].values  # Lateral position
	longitude_pos = frame_data['Local_Y'].values  # Longitude position
	vehicle_id = frame_data['Vehicle_ID'].values  # Vehicle ID
	vehicle_neighbors = frame_data['Grid_Neighbors'].values # Neighbors ID
	random_choice_track_id = np.random.choice(vehicle_id)
	length = frame_data['v_Length'].values  # Vehicle length
	width = frame_data['v_Width'].values  # Vehicle width
	vehicle_class = frame_data['v_Class'].values  # Vehicle class

	vehicle_pred_x = None
	vehicle_pred_y = None
	if 'Predicted_X' in frame_data.columns and 'Predicted_Y' in frame_data.columns:
		vehicle_pred_x = frame_data['Predicted_X'].values
		vehicle_pred_y = frame_data['Predicted_Y'].values

	# Set title
	ax.set_title(f'NGSIM trajectories - frame: {int(frame_data["Frame_ID"].iloc[0])}')

	# Plot road boundaries
	ax.plot(x_limit, [-60, -60], color='red', linestyle='--')
	ax.plot(x_limit, [60, 60], color='red', linestyle='--')

	# Plot vehicle bounding boxes based on vehicle class
	for i in range(len(frame_data)):
		# Create bounding box for each vehicle
		bounding_box = [longitude_pos[i] - length[i] / 2, lateral_pos[i] - width[i] / 2, length[i], width[i]]
		
		if vehicle_class[i] == 1:
			color = 'orange'  # Motorcycle
		elif vehicle_class[i] == 2:
			color = 'blue'  # Auto
		else:
			color = 'green'  # Truck
		
		rect = patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2], bounding_box[3],
								linewidth=1, edgecolor=color, facecolor=color)
		ax.add_patch(rect)

		future_data = df[(df['Vehicle_ID'] == vehicle_id[i]) &
						 (df['Frame_ID'] >= frame) &
						 (df['Frame_ID'] <= frame + 10)]
		
		if not future_data.empty:
			future_x = future_data['Local_X'].values
			future_y = future_data['Local_Y'].values
			ax.plot(future_y, future_x, linestyle='-', marker='o', color='purple', markersize=3, alpha=0.7, label='Ground truth')

		if vehicle_pred_x is not None and vehicle_pred_y is not None:
			pred_x = vehicle_pred_x[i]  # X coordinates for 10 future steps
			pred_y = vehicle_pred_y[i]  # Y coordinates for 10 future steps
			ax.plot(longitude_pos[i]+pred_y, lateral_pos[i]+pred_x, linestyle='--', marker='x', color='red', markersize=3, alpha=0.7, label='Predicted')
		
		# Add vehicle id to each vehicle
		ax.text(longitude_pos[i] - 2 * length[i] / 3, lateral_pos[i], str(int(vehicle_id[i])),
				color='blue', fontsize=8, clip_on=True)
		
		# select vehicle id == 22 for tracking neighbor
		if config['visualization']['enable_neighbors']:
			for neighbor in vehicle_neighbors[i]:
				for j in range(len(frame_data)):
					if vehicle_id[j] == neighbor:
						ax.plot([longitude_pos[i], longitude_pos[j]], [lateral_pos[i], lateral_pos[j]], color='black', linestyle='--')
	
	# Custom legend for vehicle classes
	legend_patches = [patches.Patch(color='orange', label='Motorcycle'),
					patches.Patch(color='blue', label='Auto'),
					patches.Patch(color='green', label='Truck'),
					Line2D([0], [0], color='purple', linestyle='-', marker='o', markersize=5, label='Ground Truth'),
					Line2D([0], [0], color='red', linestyle='--', marker='x', markersize=5, label='LSTM Predicted Trajectory')]
	ax.legend(handles=legend_patches, loc='upper right', fontsize=14)

	# Set limits and labels
	ax.set_xlim()
	ax.set_ylim(y_limit)
	ax.set_xlabel('Longitude (feet)')
	ax.set_ylabel('Lateral (feet)')
	
	# Invert x-axis
	ax.invert_xaxis()
	ax.grid(True)

	if frame == frames[-1]:
		ani.event_source.stop()  # Stop the animation
		plt.close(fig)

def plot_intersection_frame(frame, ax, df, config, frames, ani, fig):
	ax.clear()
	# loading parameter from yaml file
	x_limit1 = config['visualization']['intersection']['x_limit1']
	x_limit2 = config['visualization']['intersection']['x_limit2']
	y_limit = config['visualization']['intersection']['y_limit']
	section_limits = config['visualization']['intersection']['section_limits']

	frame_data = df[(df['Frame_ID'] == frame) &
					(df['Local_Y'] >= section_limits[0]) &
					(df['Local_Y'] <= section_limits[1])]

	if frame_data.empty:
		return

	# Get needed fields
	lateral_pos = frame_data['Local_X'].values  # Lateral position
	longitude_pos = frame_data['Local_Y'].values  # Longitude position
	vehicle_id = frame_data['Vehicle_ID'].values  # Vehicle ID
	vehicle_neighbors = frame_data['Grid_Neighbors'].values # Neighbors ID
	vehicle_directions = frame_data['Direction'].values
	vehicle_lane_id = frame_data['Lane_ID'].values
	vehicle_road_section = frame_data['Section_ID'].values
	vehicle_theta = frame_data['Theta'].values
	length = frame_data['v_length'].values  # Vehicle length
	width = frame_data['v_Width'].values  # Vehicle width
	vehicle_class = frame_data['v_Class'].values  # Vehicle class

	vehicle_pred_x = None
	vehicle_pred_Y = None
	if 'Predicted_X' in frame_data.columns and 'Predicted_Y' in frame_data.columns:
		vehicle_pred_x = frame_data['Predicted_X'].values
		vehicle_pred_y = frame_data['Predicted_Y'].values

	# Set title
	ax.set_title(f'NGSIM trajectories - frame: {int(frame_data["Frame_ID"].iloc[0])}')

	# Plot intersection boundaries
	ax.plot(x_limit1, [-60, -60], color='red', linestyle='--')
	ax.plot(x_limit2, [-60, -60], color='red', linestyle='--')
	ax.plot(x_limit1, [60, 60], color='red', linestyle='--')
	ax.plot(x_limit2, [60, 60], color='red', linestyle='--')

	ax.plot([220, 220], [-150, -60], color='red', linestyle='--')
	ax.plot([220, 220], [60, 250], color='red', linestyle='--')

	ax.plot([120, 120], [-150, -60], color='red', linestyle='--')
	ax.plot([120, 120], [60, 250], color='red', linestyle='--')

	# Plot vehicle bounding boxes based on vehicle class
	for i in range(len(frame_data)):
		# Create bounding box for each vehicle
		bounding_box = [longitude_pos[i] - length[i] / 2, lateral_pos[i] - width[i] / 2, length[i], width[i]]
		
		if vehicle_class[i] == 1:
			color = 'orange'  # Motorcycle
		elif vehicle_class[i] == 2:
			color = 'blue'  # Auto
		else:
			color = 'green'  # Truck
		
		rect = patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2], bounding_box[3],
								linewidth=1, edgecolor=color, facecolor=color, angle=90-vehicle_theta[i], rotation_point='center')
		ax.add_patch(rect)
		
		# Add vehicle id to each vehicle
		# ax.text(longitude_pos[i] - 2 * length[i] / 3, lateral_pos[i], str(int(vehicle_id[i])),
		#         color='blue', fontsize=8, clip_on=True)
		# ax.text(longitude_pos[i] - 2 * length[i] / 3, lateral_pos[i], f"X: {lateral_pos[i]:.2f}, Y: {longitude_pos[i]:.2f}",
		#         color='blue', fontsize=8, clip_on=True)
		ax.text(longitude_pos[i] - 2 * length[i] / 3, lateral_pos[i], f"Lane: {vehicle_directions[i]}",
				color='blue', fontsize=8, clip_on=True)
		
		# select vehicle id == 22 for tracking neighbor
		if config['visualization']['enable_neighbors']:
			for neighbor in vehicle_neighbors[i]:
				for j in range(len(frame_data)):
					if vehicle_id[j] == neighbor:
						ax.plot([longitude_pos[i], longitude_pos[j]], [lateral_pos[i], lateral_pos[j]], color='black', linestyle='--')
	
	# Custom legend for vehicle classes
	legend_patches = [patches.Patch(color='orange', label='Motorcycle'),
					patches.Patch(color='blue', label='Auto'),
					patches.Patch(color='green', label='Truck')]
	ax.legend(handles=legend_patches, loc='upper right', fontsize=14)

	# Set limits and labels
	ax.set_xlim()
	ax.set_ylim(y_limit)
	ax.set_xlabel('Longitude (feet)')
	ax.set_ylabel('Lateral (feet)')
	
	# Invert x-axis
	ax.invert_xaxis()
	ax.grid(True)

	if config['visualization']['save_as_gif'] and frame == frames[-1]:
		ani.event_source.stop()  # Stop the animation
		plt.close(fig)


def data_visualization(df, config):
	"""
	Dynamic display of vehicle trajectory data
	Input:
		df: Data frame
		save_as_gif: Boolean to save the animation as a gif
		gif_filename: Filename for the gif
	Output:
		None
	"""
	# loading the parameter from yml
	save_as_gif = config['visualization']['save_as_gif']
	gif_filename = config['visualization']['gif_filename']

	visual_highway = config['visualization']['visual_highway']
	visual_intersection = config['visualization']['visual_intersection']
	
	# Get unique frames
	frames = df['Frame_ID'].unique()

	# Set up the figure for animation
	fig, ax = plt.subplots(figsize=(20, 12))

	ani = None
	
	if visual_highway:
		# Create the animation
		ani = FuncAnimation(fig, plot_highway_frame, frames=frames, interval=10, fargs=(ax, df, config, frames, ani, fig))
	elif visual_intersection:
		ani = FuncAnimation(fig, plot_intersection_frame, frames=frames, interval=10, fargs=(ax, df, config, frames, ani, fig))

	if ani is not None:
		if save_as_gif:
			# Save the animation as a gif
			prefix = "highway"
			if visual_intersection:
				prefix = "intersection"
			gif_path = os.path.join(config['visualization']['gif_path'], f'{prefix}_{gif_filename}')
			ani.save(gif_path, writer='pillow', fps=10)
			print(f"GIF saved as {gif_filename}")
		else:
			plt.show()
	else:
		print("make sure at least highway or intersection has been enabled in yaml file")

def preprocess(csv_path):
	df = pd.read_csv(csv_path)
	# data cleaning
	df = df.dropna()
	# Clear different vehicle tracks of the same vehicle at the same time
	df = df.drop_duplicates(subset=['Vehicle_ID', 'Global_Time'])

	df = df.sort_values(by=['Vehicle_ID', 'Global_Time'], ascending=[True, True])

	return df

def main(args):
	dataset_path = args.dataset

	# csv_path = os.path.join(dataset_path, US101_folder, US101_time1_csv)

	# data cleaning, sorting
	# sorted_df = preprocess(csv_path)

	# # select part of data for visualization
	# # just only keep 300 frame for visualization
	# visual_df = sorted_df.loc[(sorted_df['Frame_ID'] >= 0) & (sorted_df['Frame_ID'] < 300)]

	# data_visualization(visual_df,save_as_gif=False)

	"""Batch Processing"""
	config = load_configuration("preprocess_setting.yaml")

	all_csv_path, scenarios = load_all_csv_path(config)

	if config['enable_preprocess']:
		# cleaning
		df_dict = BatchData_Load_clearning(scenarios, all_csv_path)
		feature_process = MultiThread_FeatureProcess(df_dict, config)
		feature_process.run()
	if config['enable_visualization']:
		# don't need to run data processing, just loading the data from pkl
		data = {}
		for name in scenarios:
			# if name in ['peachtree']:
			pkl_path = os.path.join(config['output']['pickle_output_folder'], f'{name}_valid_data.pkl')
			data[name] = read_pickle(pkl_path)
		
		# for highway scenario
		if config['visualization']['visual_highway']:
			data_visualization(data["us101_time1"], config)

		# for intersection scenario
		if config['visualization']['visual_intersection']:
			intersection_dict = {}
			intersection_visual_df = data['peachtree']

			print(intersection_visual_df)

			# Get vehicles at the intersection
			for intersection_id, group in intersection_visual_df.groupby('Section_ID'):
				intersection_dict[intersection_id] = group.sort_values('Global_Time')

			# select the first intersection for visualization
			visual_intersection_id = list(intersection_dict.keys())[:2]
			visual_intersection_df = intersection_visual_df[(intersection_visual_df['Section_ID'].isin(visual_intersection_id) ) | (intersection_visual_df['Int_ID'].isin(visual_intersection_id))]

			# cutting a part for visualization
			visual_intersection_df = visual_intersection_df[(visual_intersection_df['Frame_ID'] > 8200) & (visual_intersection_df['Frame_ID'] < 8500)]
			data_visualization(visual_intersection_df, config)
	
	if config['enable_LSTM_training']:

		lstm_config = load_configuration("./LSTM/LSTM_setting.yaml")

		if lstm_config['combine_us101']:
			# combine US101
			prefix_lst = ['train', 'test', 'valid']
			for prefix in prefix_lst:
				combine_US101_data(lstm_config, prefix)
		elif lstm_config['combine_all']:
			# combine all dataset
			prefix_lst = ['train', 'test', 'valid']
			for prefix in prefix_lst:
				combine_all_data(lstm_config, prefix)
		elif lstm_config['enable_training']:
			# initialize the tensorboard to save training process
			saved_tensorboard_path = os.path.join('summary')
			os.makedirs(saved_tensorboard_path, exist_ok=True)
			writer = SummaryWriter(saved_tensorboard_path)

			# for checkpoints
			saved_ckpt_path = os.path.join(lstm_config['saved_ckpt_path'])
			os.makedirs(saved_ckpt_path, exist_ok=True)

			# train(lstm_config, writer, 'US101')

			# train(lstm_config, writer, 'I80')

			train(lstm_config, writer, 'peachtree')
	if config['enable_lstm_evaluation']:
		lstm_config = load_configuration("./LSTM/LSTM_setting.yaml")
		dataset_path = lstm_config['dataset_path']

		for dataset_name in ['US101', 'I80', 'peachtree']:
			if dataset_name == 'I80':
				evaluation(dataset_name, lstm_config)

	if config['enable_lstm_visualization']:
		
		lstm_config = load_configuration("./LSTM/LSTM_setting.yaml")
		dataset_path = lstm_config['dataset_path']

		data = {}
		spec_file = 'pkl_file'
		for dataset_name in ['US101', 'I80', 'peachtree']:
			if dataset_name == 'US101':
				spec_file = 'time1_file'
			# if dataset_name == 'US101':
			if not os.path.exists(config['prediction_df_path'][dataset_name]):
				pkl_path = os.path.join(dataset_path, lstm_config['dataset']['test'][dataset_name][spec_file])
				df = read_pickle(pkl_path)
				data[dataset_name] = evaluation(dataset_name, lstm_config, df, spec_file)

				prediction_path = os.path.join(dataset_path, f'{dataset_name}_prediction_visual_data.pkl')

				write_pickle(data[dataset_name], prediction_path)
			else:
				data[dataset_name] = read_pickle(os.path.join(config['prediction_df_path'][dataset_name]))

		if config['visualization']['visual_highway']:
			us101_visual_df = data['US101'][(data['US101']['Frame_ID'] >=7200 ) & (data['US101']['Frame_ID'] < 7500)]
			data_visualization(us101_visual_df, config)

		




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# got LNF or LT from input
	parser.add_argument('--dataset', \
						default='~/hw/cs7641/project/Vehicle-Prediction/dataset',
						help='the root path of dataset')
	
	# parser.add_argument('--experiment_idx',\
	#                     default='1',
	#                     help='select which experiment you want to run')
	
	args = parser.parse_args()
	main(args)
