import pandas as pd
import numpy as np
from haversine import haversine, Unit
import argparse

DEFAULTPATH = 'trip_duration_proj/data/nyc_taxi_trip_duration/test.parquet'


def convert_type(data:pd.DataFrame, info:dict) :
    for col, type in info.items() :
        data[[*col]] = data[[*col]].astype(type)
    
    # replace back the arbitrary value (100) to the smallest value in each column
    for col in ['snow depth', 'precipitation'] :
        val = sorted(data[col].unique())[1]
        data[col] = data[col].replace(100, val)


def extract(data:pd.DataFrame) :
    data['day'] = data['pickup_datetime'].dt.day 
    data['weekday'] = data['pickup_datetime'].dt.day_name()
    data['hour'] = data['pickup_datetime'].dt.hour
    data['month'] = data['pickup_datetime'].dt.month

    data['rush_hour'] = data['hour'].isin(range(7, 19)).map({True:1, False:0})

    def calculate_distances(row:pd.Series):
        # haversine distance
        pickup = (row['pickup_latitude'], row['pickup_longitude'])
        dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
        haversine_dist = haversine(pickup, dropoff, unit=Unit.KILOMETERS)
        
        # manhattan distance
        manhattan_dist = abs(pickup[0] - dropoff[0]) * 111 + abs(pickup[1] - dropoff[1]) * 85

        return pd.Series({
            'haversine_distance': haversine_dist,
            'manhattan_distance': manhattan_dist
        })
    
    data[['haversine_distance', 'manhattan_distance']] = data.apply(calculate_distances, axis=1)



def remove_outliers(data:pd.DataFrame) :
    # first, remove outliers based on the trip duration
    data = data[(data['trip_duration'] < 10000) & data['trip_duration'] > 120]

    # then, remove outliers based on the coordinates
    latitude = 40.4774, 40.9176
    longitude = -74.2591, -73.7004

    data = data[
    (data['pickup_latitude'].between(latitude[0], latitude[1])) &
    (data['pickup_longitude'].between(longitude[0], longitude[1])) &
    (data['dropoff_latitude'].between(latitude[0], latitude[1])) &
    (data['dropoff_longitude'].between(longitude[0], longitude[1]))
    ]

    # remove row of passengeer_count = 0
    data = data[data['passenger_count'] > 0]


    # this functions combines all the above functions to prepare the data for modeling 
def prepare(data:pd.DataFrame) :
    for col in ['snow depth', 'snow fall', 'precipitation'] :
        data[col] = data[col].replace('T', 100)

    conversion_info = {('vendor_id' , 'passenger_count') : 'category',
                        ('precipitation', 'snow depth') : 'float'}
    convert_type(data, conversion_info)
    extract(data)
    remove_outliers(data)
    
    # drop the columns that are not needed for the model
    to_drop = ['id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag', 'maximum temperature', 'minimum temperature', 'snow fall']
    data.drop(columns= to_drop, inplace=True)

    data.to_parquet('trip_duration_proj/prepared_data/test.parquet')

    return data


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description= 'get the path of the data for the script "prepare_data.py"')
    parser.add_argument('-p', '--path', type=str, default=DEFAULTPATH, help='The path of the data')
    args = parser.parse_args()

    data = pd.read_parquet(args.path)

    data = prepare(data)




