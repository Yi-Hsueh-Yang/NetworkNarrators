import pandas as pd
import re
import time

from data_collection.kafka_collection import datacollection

def test_datacollection():
    # Call the function with a collection time of 2 seconds
    collection_time = 2
    df = datacollection(collection_time)

    # Check if the DataFrame has the correct columns and types
    assert len(df.columns) == 3
    assert list(df.columns) == ['timestamp', 'userid', 'request']
    assert df['userid'].dtype == int

