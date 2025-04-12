import numpy as np
from geopy.geocoders import Nominatim #Used for geocoding - takes name of the city and gives coordinates
import pytz # time zones
from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
import os

from meteostat.series.aggregate import aggregate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from librosa.util import frame

raw_data_dir = "../data/raw"
processed_data_dir = "../data/processed/"

os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)


data_file_csv = os.path.join(raw_data_dir, "multicity_weather.csv")
data_file_segmented = os.path.join(processed_data_dir, "segmented_data.npz")

if os.path.exists(data_file_csv):
    print(f"Reading data from file {data_file_csv}")
    aggregate_data = pd.read_csv(data_file_csv)
else:
    print("Downloading necessary data")

    cities = {
        "Kraków": None,
        "Szczebrzeszyn": None,
        "Lublin": None,
        "Gdańsk": None,
        "Szczecin": None,
        "Poznań": None,
        "Łódź": None,
        "Warszawa": None,
        "Wrocław": None,
    }

    geolocator = Nominatim(user_agent="weather_app")

    #start - start date for downloadinng data
    #end - end date - actual time in time zone Europe/Warsaw
    #.replace(tzinfo=None) - removes the time zone information to operate on local time.
    start = datetime(2021, 11, 1).replace(tzinfo=None)
    end = datetime.now(pytz.timezone("Europe/Warsaw")).replace(tzinfo=None)

    cities_data = []

    for city in cities.keys():
        try:
            # Creates a Point object from the meteostat library that represents a specific location based on geographic coordinates
            geo_location = geolocator.geocode(city)
            location_coords = Point(geo_location.latitude, geo_location.longitude)
            meteo_data = Hourly(location_coords, start, end).fetch()

            meteo_data["city"] = city
            meteo_data["lat"] = geo_location.latitude
            meteo_data["lon"] = geo_location.longitude
            cities_data.append(meteo_data)
        except Exception as e:
            print(f"Error during downloading data for city {city}: {e}")


    # Aggregating data to one DF
    aggregate_data = pd.concat(cities_data)
    aggregate_data.reset_index(inplace=True)

    aggregate_data["hour"] = aggregate_data["time"].dt.hour
    aggregate_data["day"] = aggregate_data["time"].dt.day
    aggregate_data["month"] = aggregate_data["time"].dt.month
    aggregate_data["dayofweek"] = aggregate_data["time"].dt.dayofweek

    le = LabelEncoder()
    aggregate_data["city_encoded"] = le.fit_transform(aggregate_data["city"])

    aggregate_data.drop(columns=["prcp", "snow", "tsun"], inplace=True, errors="ignore")
    aggregate_data.fillna(aggregate_data.median(numeric_only=True), inplace=True)

    # print(aggregate_data.shape)
    # print(aggregate_data.isna().sum())

    all_numeric_columns = aggregate_data.select_dtypes(include=["float64", "int64"]).columns.values
    columns_without_temp = np.delete(all_numeric_columns, np.where(all_numeric_columns == "temp"))

    # Scaler for input parameters
    scaler_input = StandardScaler()
    aggregate_data[columns_without_temp] = scaler_input.fit_transform(aggregate_data[columns_without_temp])

    # Scaler for output parameters
    scaler_output = StandardScaler()
    aggregate_data[["temp"]] = scaler_output.fit_transform(aggregate_data[["temp"]])


    # Deleting unnecessary columns
    columns_to_drop = ["time", "city", "lat", "lon"]
    aggregate_data.drop(columns_to_drop, inplace=True, errors="ignore")

    # Sorting data after "time" column - chronological order - VERY IMPORTANT
    aggregate_data.sort_values("time", inplace=True)
    aggregate_data.reset_index(drop=True, inplace=True)

    aggregate_data.to_csv(data_file_csv, index=False)

if os.path.exists(data_file_segmented):
    print("All necessary data are prepared")
else:
    print("Segmenting data")

    #DATA SEGMENTATION
    number_of_days = 7
    hours = 24
    input_data_length = number_of_days * hours
    data_to_predict_length = 24

    np_aggregate_data = np.matrix(aggregate_data)
    #print("Data shape before segmentation: ", np_aggregate_data.shape)

    input_segmented_data = frame(np_aggregate_data, frame_length=input_data_length + data_to_predict_length, hop_length=1, axis=0)

    #print('Data shape after segmentation:', input_segmented_data.shape)

    #We will do the same with the output signal, but first we have to shift it to synchronize the data (prediction of the future with current input).
    # We will shift the output signal by the length of the input signal window so that the beginnings of both windows start at the same place.
    temp_data_shifted = aggregate_data['temp'].shift(-input_data_length)

    np_temp_data_shifted = np.array(temp_data_shifted)
    output_segmented_data = frame(np_temp_data_shifted, frame_length=data_to_predict_length, hop_length=1, axis=0)
    #print(output_segmented_data.shape)

    output_segmented_data = output_segmented_data[:input_segmented_data.shape[0], :]
    # print(np.isnan(input_segmented_data).any())
    # print(np.isnan(output_segmented_data).any())

    #print(output_segmented_data[-data_to_predict_length:, :])

    input_segmented_data = input_segmented_data[:-data_to_predict_length, :, :]
    output_segmented_data = output_segmented_data[:-data_to_predict_length, :]

    np.savez(data_file_segmented, X=input_segmented_data, y=output_segmented_data)