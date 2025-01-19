import pandas as pd

taxi_zones = pd.read_csv("../data/taxi_zone_lookup_coordinates.csv")

def preprocess(data):
    data['duration'] = data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']
    data['duration'] = data['duration'].apply(lambda x: x.total_seconds() / 60)
    data = data[(data['duration'] >= data['duration'].quantile(0.04)) & (data['duration'] <= data['duration'].quantile(0.98))]
    categorical = ['trip_type', 'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = ['duration']
    data = data[numerical + categorical + target]
    data['trip_type'].fillna(data.trip_type.mode().item(), inplace=True)
    data = pd.merge(data, taxi_zones[['LocationID', 'latitude', 'longitude']], left_on='PULocationID', right_on='LocationID', how='left').rename(columns={"latitude": "PULat",
                                                                                                                                                   "longitude": "PULon"})
    data = pd.merge(data, taxi_zones[['LocationID', 'latitude', 'longitude']], left_on='DOLocationID', right_on='LocationID', how='left').rename(columns={"latitude": "DOLat",
                                                                                                                                                   "longitude": "DOLon"})
    data = data.drop(columns=["PULocationID", "DOLocationID", "LocationID_x", "LocationID_y"])
    data["PULat"].fillna(data["PULat"].mean(), inplace=True)
    data["PULon"].fillna(data["PULon"].mean(), inplace=True)
    data["DOLat"].fillna(data["DOLat"].mean(), inplace=True)
    data["DOLon"].fillna(data["DOLon"].mean(), inplace=True)
    onehot = pd.get_dummies(data['trip_type'], prefix="triptype").map(lambda x: 1 if x else 0)
    data = data.join(onehot).drop(columns=['trip_type'])
    data = data[(data['trip_distance'] >= data['trip_distance'].quantile(0.04)) & (data['trip_distance'] < data['trip_distance'].quantile(0.99))]
    
    for column in ['PULat', 'PULon', 'DOLat', "DOLon"]:
        data[column] = (data[column] - data[column].mean()) / data[column].std()

    for column in ['trip_distance', 'duration']:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    
    return data[['PULat', 'PULon', 'DOLat', "DOLon", "trip_distance"]], data['duration']