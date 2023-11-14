import logging

import azure.functions as func
import requests

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
@app.queue_output(
    arg_name="msg",
    queue_name="data-cute",
    # connection="AzureWebJobsStorage"
)
def etl_azure_function(myTimer: func.TimerRequest) -> None:
    logging.info("Python timer trigger function ran at %s", myTimer)
    telraam_data = requests.get(
        telraam_url, headers={"X-API-Key": telraam_api_key}
    ).json()
    community_sensor_data = requests.get(community_sensor_url).json()
    tansformer = Transformer()
    transformed_data = tansformer.transform(telraam_data, community_sensor_data)
    logging.info("Transformed data Shape: %s", transformed_data.shape)
    return transformed_data
