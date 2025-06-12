# <span style=color:#5CB338>NYC-Taxi-Trip-Duration-Project</span>

## Overview
This project is based on [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) competition on kaggle.\
The aim of this project is predicting the taxi trip duration in New York city.

The data used in it is the official competition data on kaggle in addition to a weather conditions data for New York at the same period.
Which I added for gaining more information that can help.

## ➡️ Usage
```
cd NYC-Taxi-Trip-Duration-Project
python train.py
```
## ➡️ To install the requirements file
```
pip install -r requirements.txt
```

## Note :
The data files used in this project are all in `parquet` format for efficient and fast storing and reading.\
You need to install one of these two libraries to read and use parquet files with pandas.
```
# only one is needed
pip install pyarrow
# pip install fastparquet
```
## Directories Structure
```
├── README.md
├── requirements.txt
├── EDA.ipynb
├── project_report.pdf
├── prepare_data.py
├── train.py
├── prepared_data
│   ├── train.parquet
│   └── test.parquet
│
└── data
    └── nyc_taxi_trip_duration
    │   ├── train.parquet
    │   └── val.parquet
    │
    └── weather_data_nyc_centralpark_2016.parquet
```
## Files Description

* `requirements.txt` : Contains the libraries and packages you need to run the project locally.
* `EDA.ipynb` : The exploratory data analysis (EDA) on the training data.
* `project_report.pdf` : A report were I summarized the findings of EDA, changes made to the data, and the final modeling results.
* `prepare_data.py` : Python file for preparing the data before processing it.\
The _prepare()_ function in the file combines all the functions in the file to run the preparation steps on the test data. It's almost the same modifications occured on the training data (adding features, converting types,...).
* `train.py` : The training script of the models. This file contains the pipelines and processing steps I used on the data.
* `prepared_data` : a directory contains the prepared data (the data after applying the prepare_data script on it)
* `data` : The directory containing the original data
  * `nyc_taxi_trip_duration` : The original kaggle competition data splitted into train and test datasets.
  * `weather_data_nyc_centralpark_2016.parquet` : The weather data of New York city.
