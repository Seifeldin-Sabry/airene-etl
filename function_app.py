import logging
from math import asin, cos, radians, sin, sqrt

import azure.functions as func
import numpy as np
import pandas as pd
import requests

app = func.FunctionApp()


# URLS and API KEYS #
telraam_url = "https://telraam-api.net/v1/reports/traffic_snapshot_live"
telraam_api_key = "R0bdepMpRxP9MPVWSOPr9OGRFseK4Ov59NTa8bz5"
community_sensor_url = "https://data.sensor.community/static/v2/data.1h.json"
weather_api_key = "f70225a3abc24e8a9e1104007231411"
#######################


@app.schedule(
    schedule="0 0 * * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False
)
@app.service_bus_topic_output(
    topic_name="data-cute",
    arg_name="message",
    queue_name="data-cute",
    connection="AzureServiceBusConnectionString",
)
def etl_azure_function(myTimer: func.TimerRequest, message: func.Out[str]):
    logging.info("Python timer trigger function ran at %s", myTimer)
    print("Python timer trigger function ran at %s", myTimer)
    try:
        logging.info("Fetching data...")
        telraam_data = requests.get(
            telraam_url, headers={"X-API-Key": telraam_api_key}
        ).json()
        community_sensor_data = requests.get(community_sensor_url).json()
        logging.info("Data fetched")
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return

    transformer = Transformer()
    transformed_data = transformer.transform(telraam_data, community_sensor_data)
    json = transformed_data.to_json(orient="records")
    logging.info("Transformed data Shape: %s", transformed_data.shape)
    print("Transformed data Shape: %s", transformed_data.shape)
    message.set(json)


class Transformer:
    def __init__(self):
        pass

    def transform(self, base_data, sensor_data):
        tldf = pd.DataFrame(base_data["features"])
        csdf = pd.DataFrame(sensor_data)
        logging.info("Cleaning data...")
        tldf = self.clean_telraam_data(tldf)
        csdf = self.clean_community_sensor_data(csdf)
        logging.info("Calculating distance matrix...")
        merged = self.calculate_and_merge_based_on_distance(tldf, csdf)
        logging.info("Adding weather data...")
        with_weather = self.add_weather_date(merged)
        return with_weather

    def clean_telraam_data(self, tldf):
        # Explode the properties and geometry columns
        tldf = pd.concat(
            [tldf.drop(["properties"], axis=1), tldf["properties"].apply(pd.Series)],
            axis=1,
        )
        tldf = pd.concat(
            [tldf.drop(["geometry"], axis=1), tldf["geometry"].apply(pd.Series)], axis=1
        )

        # Clean up NaNs
        tldf = tldf.replace("", np.nan)
        tldf = tldf.dropna().drop(
            ["type", "period", "uptime", "bike", "pedestrian"], axis=1
        )
        tldf = tldf.drop(["last_data_package"], axis=1)
        tldf = tldf.reset_index(drop=True)

        return tldf

    def clean_community_sensor_data(self, csdf):
        csdf_dropped = csdf.drop(
            ["location", "sensor", "sensordatavalues", "sampling_rate"], axis=1
        )
        expanded_location = csdf["location"].apply(pd.Series)
        expanded_sensor = csdf["sensor"].apply(pd.Series)
        expanded_sensordatavalues = csdf["sensordatavalues"].apply(pd.Series)

        csdf = pd.concat(
            [
                csdf_dropped,
                expanded_location,
                expanded_sensor,
                expanded_sensordatavalues,
            ],
            axis=1,
        )
        # Change sensor names
        csdf.columns = [
            f"sensor {str(x)}" if isinstance(x, int) else x for x in csdf.columns
        ]
        csdf = csdf.drop(["exact_location", "indoor", "pin"], axis=1)
        # Drop all "id" columns
        csdf = csdf.drop([col for col in csdf.columns if "id" in col], axis=1)

        # cast longitude and latitude to float
        csdf["longitude"] = csdf["longitude"].astype(float)
        csdf["latitude"] = csdf["latitude"].astype(float)
        # Only keep Belgian sensors
        csdf = csdf[csdf["country"] == "BE"]
        # Change timestamp to datetime
        csdf["timestamp"] = pd.to_datetime(csdf["timestamp"]).dt.round("H")
        csdf = csdf.reset_index(drop=True)

        return csdf

    # https://stackoverflow.com/a/4913653
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # Decimal to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    def calculate_distance_matrix(self, base_df, df_to_merge):
        """
        Calculate the distance matrix between two sets of points using the Haversine formula.
        """
        distances = np.zeros((len(base_df), len(df_to_merge)))

        # TODO: Get rid of iterrows
        for i, base_row in base_df.iterrows():
            base_lon, base_lat = base_row["coordinates"][0][0]
            for j, merge_row in df_to_merge.iterrows():
                merge_lon, merge_lat = merge_row["longitude"], merge_row["latitude"]
                distances[i, j] = self.haversine(
                    base_lon, base_lat, merge_lon, merge_lat
                )

        return distances

    def calculate_and_merge_based_on_distance(self, base_df, df_to_merge):
        """
        Efficiently merge dataframes based on the distance in latitude and longitude.
        """
        # Calculate distance matrix using Haversine formula
        distance_matrix = self.calculate_distance_matrix(base_df, df_to_merge)

        # Pair the rows close to eachother
        threshold = 2  # kilometers
        idx_pairs = np.where(distance_matrix < threshold)

        # Combine the rows from both dataframes
        combined_data = []
        for i, j in zip(*idx_pairs):
            combined_row = pd.concat([base_df.iloc[i], df_to_merge.iloc[j]], axis=0)
            combined_row["distance_km"] = distance_matrix[i, j]
            combined_data.append(combined_row)

        return pd.concat(combined_data, axis=1).T

    def add_weather_date(self, df):
        # Get a list of unique segment_ids
        segment_ids = df["segment_id"].unique()

        # Loop through the segment_ids
        for segment_id in segment_ids:
            # Get the weather data for the segment_id
            try:
                latitude = df.loc[df["segment_id"] == segment_id, "latitude"].iloc[0]
                longitude = df.loc[df["segment_id"] == segment_id, "longitude"].iloc[0]
                weather_data = self.get_weather_data(latitude, longitude)

                # Add the weather data to the DataFrame
                df.loc[df["segment_id"] == segment_id, "weather_data"] = str(
                    weather_data
                )
            except Exception:
                df.loc[df["segment_id"] == segment_id, "weather_data"] = np.nan

        return df

    def get_weather_data(self, latitude, longitude):
        # Get the weather data
        parsed_endpoint = f"https://api.weatherapi.com/v1/current.json?key=9dcbfb937193476d8a4110235231811&q={latitude},{longitude}&aqi=yes"
        weather_data = requests.get(parsed_endpoint).json()
        return weather_data
