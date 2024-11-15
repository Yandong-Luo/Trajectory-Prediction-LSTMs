# Changelog
All notable changes to this project will be documented in this file.


## [0.1.1] - 2024-10-07
### Changed
- Yandong Luo: Finish the data cleaning, sorting, and visualization

## [0.1.2] - 2024-10-08
### Changed
- Yandong Luo: Fixed the issuse of visualization could not run in preprocess.ipynb

## [0.1.3] - 2024-10-16
### Changed
- Yandong Luo: The entire content of feature engineering has been preliminarily completed and processed simultaneously using a multi-threaded approach.

## [0.1.4] - 2024-10-16
### Changed
- Yandong Luo: Use a more efficient processing method for feature engineering. Divide each data set into a process, split the data of each data set, and then use multi-threading to process them in parallel.

## [0.1.5] - 2024-10-18
### Changed
- Yandong Luo: Adjust the number of threads to make the algorithm faster. Fixed the problem of failure to write .pkl files.

## [0.1.6] - 2024-10-18
### Changed
- Yandong Luo: Added visualization of neighbors in visualization. Divided the dataset into training set, validation set and test set. Added yaml file for preprocessing parameter configuration. Added theta feature. Filtered invalid trajectories with less than 3s.

## [0.1.7] - 2024-10-20
### Changed
- Yandong Luo: Added intersection visualization and verify it. All preprocess are basically complete.

## [0.1.8] - 2024-10-20
### Changed
- Yandong Luo: update README for intersection visualization and link to the preprocessed dataset

## [1.1.1] - 2024-11-01
### Changed
- Yandong Luo: Complete the processing of pkl file to train input. The processing includes loading data for pytorch in the form of dataloader and obtaining vehicle history, future trajectory, and historical trajectory of neighbor vehicles. Add regularization to the preprocessing part (but I still don’t use regularized data, I don’t think it’s needed)

## [2.1.1] - 2024-11-01
### Changed
- Cari He: Merged linear regression, code in LinearRegression.ipynb

## [1.1.2] - 2024-11-02
### Changed
- Yandong Luo: Complete the training code and network structure. The training process is consistent with expectations. The model is converging.

## [1.1.3] - 2024-11-05
### Changed
- Yandong Luo: The validation process is added during training, and the learning rate is optimized using OneCycleLR.

## [1.1.4] - 2024-11-06
### Changed
- Yandong Luo: Complete LSTM visualization and training of all datasets.

## [1.1.5] - 2024-11-07
### Changed
- Yandong Luo: Complete the evaluation for LSTM.

## [1.1.5] - 2024-11-10
### Changed
- Yandong Luo: Adjust LSTM training parameters
