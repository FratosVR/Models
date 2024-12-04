from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os
import numpy as np


class DataLoader:
    def __init__(self):
        self.__api = KaggleApi()
        self.__api.authenticate()
        self.__dataset_path = "./dataset"
        self.__api.dataset_download_cli(
            "alk222/csv-pose-animations", path=self.__dataset_path, unzip=True)

    def dataset_loader(self, time: float) -> pd.DataFrame:
        list_data = []

        for file in os.listdir(self.__dataset_path):
            try:
                list_data.append((file.split(" ")[0], pd.read_csv(
                    os.path.join(self.__dataset_path, file), delimiter=",", dtype=np.float32)))
            except Exception as e:
                print(f"Error reading {file}")
                print(e)
                continue
        max_length = max(map(lambda x: x[1].shape[1], list_data))
        list_data = [x.reindex(range(max_length), fill_value=0)
                     for x in list_data]
        return list_data
