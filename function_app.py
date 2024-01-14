import asyncio
import logging
import time

import azure.functions as func
import numpy as np
import pandas as pd
import requests

app = func.FunctionApp()

WEATHER_API_RETRIES = 1
THRESHOLD = 2  # kilometers


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
async def etl_azure_function(myTimer: func.TimerRequest, message: func.Out[str]):
    # Start timer
    start = time.time()
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
    transformed_data = await transformer.transform(telraam_data, community_sensor_data)
    json = transformed_data.to_json(orient="records")
    logging.info("Transformed data Shape: %s", transformed_data.shape)
    print("Transformed data Shape: %s", transformed_data.shape)
    print("TIME TAKEN: ", time.time() - start)
    message.set(json)


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        logging.info(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        return result

    return wrapper


class Transformer:
    def __init__(self):
        pass

    @timer
    async def transform(self, base_data, sensor_data):
        tldf = pd.DataFrame(base_data["features"])
        csdf = pd.DataFrame(sensor_data)
        logging.info("Cleaning data...")
        tldf = self.clean_telraam_data(tldf)
        csdf = self.clean_community_sensor_data(csdf)
        logging.info("Calculating distance matrix...")
        merged = self.calculate_and_merge_based_on_distance(tldf, csdf)
        logging.info("Adding weather data...")
        with_weather = await self.add_weather_data(merged)
        return with_weather

    @timer
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

    @timer
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

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Vectorized Haversine formula to calculate distances between two sets of points.
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    @timer
    def calculate_distance_matrix(self, base_df, df_to_merge):
        """
        Calculate the distance matrix between two sets of points using the Haversine formula.
        """
        base_coords = np.array(base_df["coordinates"].apply(lambda x: x[0][0]).tolist())
        merge_coords = df_to_merge[["longitude", "latitude"]].to_numpy()

        # Expanding base_coords and merge_coords to 3D for broadcasting
        base_coords_expanded = base_coords[:, np.newaxis, :]
        merge_coords_expanded = merge_coords[np.newaxis, :, :]

        # Calculate distances
        distances = self.haversine(
            base_coords_expanded[..., 0],
            base_coords_expanded[..., 1],
            merge_coords_expanded[..., 0],
            merge_coords_expanded[..., 1],
        )

        return distances

    def calculate_and_merge_based_on_distance(self, base_df, df_to_merge):
        """
        Efficiently merge dataframes based on the distance in latitude and longitude.
        """
        # Calculate distance matrix using Haversine formula
        distance_matrix = self.calculate_distance_matrix(base_df, df_to_merge)

        # Pair the rows close to each other
        idx_pairs = np.where(distance_matrix < THRESHOLD)

        # Combine the rows from both dataframes
        combined_data = []
        for i, j in zip(*idx_pairs):
            combined_row = pd.concat([base_df.iloc[i], df_to_merge.iloc[j]], axis=0)
            combined_row["distance_km"] = distance_matrix[i, j]
            combined_data.append(combined_row)

        return pd.concat(combined_data, axis=1).T

    @timer
    async def add_weather_data(self, df):
        segment_ids = df["segment_id"].unique()
        tasks = [self.fetch_weather_data(segment_id, df) for segment_id in segment_ids]

        weather_data_results = await asyncio.gather(*tasks)

        # Update DataFrame after gathering all results
        for segment_id, weather_data in weather_data_results:
            df.loc[df["segment_id"] == segment_id, "weather_data"] = weather_data

        return df

    async def fetch_weather_data(self, segment_id, df, retry=WEATHER_API_RETRIES):
        try:
            latitude = df.loc[df["segment_id"] == segment_id, "latitude"].iloc[0]
            longitude = df.loc[df["segment_id"] == segment_id, "longitude"].iloc[0]
            weather_response = requests.get(
                f"https://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={latitude},{longitude}&aqi=yes"
            )
            weather_data = weather_response.json()
            if weather_data is None:
                print(weather_response)
                if retry > 0:
                    self.fetch_weather_data(segment_id, df, retry - 1)
                raise Exception(
                    f"Weather data is None for segment {segment_id} latitude {latitude} longitude {longitude}"
                )
            return segment_id, str(weather_data)
        except Exception as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return segment_id, None