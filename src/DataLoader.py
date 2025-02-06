from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm


def flatten(xss):
    return [x for xs in xss for x in xs]


class DataLoader:
    def __init__(self):
        self.__api = KaggleApi()
        self.__api.authenticate()
        self.__dataset_path = "./dataset"
        if not os.path.exists(self.__dataset_path):
            self.__api.dataset_download_cli(
                "alk222/csv-pose-animations", path=self.__dataset_path, unzip=True)

    def data_cleaner(self, data: list[tuple[int, np.ndarray]], max_length: int) -> list[tuple[int, np.ndarray]]:
        new_elems = []
        usable_data = []
        for label, row in tqdm(data):
            n_row = row.shape[0]
            if n_row < 10:
                continue
            if n_row == max_length:
                usable_data.append((label, row))
            elif n_row > max_length:
                # Getting the elements from 90 to the end
                ex = row.copy()[max_length + 1:-1]
                # Capping the row to 90 elems
                row = row[:max_length].copy()
                usable_data.append((label, row))
                if ex.shape[0] > 19:
                    new_elems.append((label, ex))
            elif n_row < max_length:
                n_row_aux = n_row
                row_aux = row.copy()
                row_aux = np.repeat(
                    row_aux, [2 for _ in range(n_row_aux)], axis=0)
                if row_aux.shape[0] > max_length:
                    row_aux = row_aux[:max_length].copy()
                    usable_data.append((label, row_aux))
                else:
                    new_elems.append((label, row_aux))
        del data
        if not (len(new_elems) == 0):
            cleaned_data = self.data_cleaner(new_elems, max_length)
            for cdl, cdr in cleaned_data:
                usable_data.append((cdl, cdr))

        return usable_data

    def dataset_loader(self) -> pd.DataFrame:
        list_data: list[tuple[string, np.ndarray]] = []

        print("Loading dataset")
        for file in tqdm(os.listdir(self.__dataset_path)):
            if not file.endswith(".csv"):
                continue
            try:
                list_data.append((file.split("_")[0], pd.read_csv(
                    os.path.join(self.__dataset_path, file), delimiter=",", dtype=np.float32).to_numpy()))
            except Exception as e:
                print(f"Error reading {file}")
                print(e)
                continue
        max_length = 90
        print(list_data[0:10])
        cleaned_data: list[tuple[int, np.ndarray]
                           ] = self.data_cleaner(list_data, max_length)
        errors: int = 0
        for label, data in tqdm(cleaned_data):
            if len(data) > 90 or len(data) < 90:
                print(f"Error with {label} with size {data.shape[0]}")
                errors += 1
        print(f"Errors: {errors} out of {len(cleaned_data)}, size mean: {
              np.mean([data.shape[0] for label, data in cleaned_data])}")

        # print(list_data)
        return list_data


if __name__ == "__main__":
    dl = DataLoader()
    dl.dataset_loader()
