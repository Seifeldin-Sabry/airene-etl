import logging

import azure.functions as func
import requests
import pandas as pd

from service.transformer import Transformer

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

    try:
        telraam_data = requests.get(
            telraam_url, headers={"X-API-Key": telraam_api_key}
        ).json()
        community_sensor_data = requests.get(community_sensor_url).json()
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return

    transformer = Transformer()
    transformed_data = transformer.transform(telraam_data, community_sensor_data)
    json = transformed_data.to_json(orient="records")
    logging.info("Transformed data Shape: %s", transformed_data.shape)
    message.set(json)
