# from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os
import numpy as np
import shutil
from tqdm import tqdm
import glob


def flatten(xss):
    """Flatten a nested list structure.

    :param xss: Nested list to flatten
    :type xss: list[list[Any]]
    :return: Flattened list
    :rtype: list[Any]
    """
    return [x for xs in xss for x in xs]


class DataLoader:
    """Dataset loader and preprocessor for pose animation data."""

    def __init__(self, dataset_path: str, interval: float = 1.0) -> None:
        """Initialize the DataLoader instance.

        :param dataset_path: Path to the dataset directory
        :type dataset_path: str
        :param interval: Data sampling interval in seconds, defaults to 1.0
        :type interval: float, optional
        """
        # self.__api = KaggleApi()
        # self.__api.authenticate()
        #: Path to the dataset directory
        self.__dataset_path: str = dataset_path
        #: Data sampling interval in seconds
        self.__interval: float = interval

    def __dataset_cleaner_aux(self, data: list[tuple[str, str, np.ndarray]], max_length: int) -> list[tuple[str, str, np.ndarray]]:
        """Clean and standardize animation data to uniform length.

        Splits elements exceeding max_length and extends shorter sequences.
        Removes elements with fewer than 10 frames. Generates new filenames
        for split sequences with "-1", "-2" suffixes.

        :param data: Original dataset as list of (label, filename, array) tuples
        :type data: list[tuple[str, str, np.ndarray]]
        :param max_length: Target number of frames per sequence
        :type max_length: int
        :return: Standardized animation sequences
        :rtype: list[tuple[str, str, np.ndarray]]
        """
        new_elems = []
        usable_data = []
        # Iterate over all elements
        for label, filename, row in tqdm(data):
            n_row = row.shape[0]
            # if the row is empty or has less than 10 frames, skip it
            if n_row < 10:
                continue
            # if the row has exactly the number of max_length frames, add it to the usable data
            if n_row == max_length:
                usable_data.append((label, filename, row))
            # if the row has more than max_length frames, split it into two rows
            elif n_row > max_length:
                # Getting the elements from 90 to the end
                ex = row.copy()[max_length + 1:-1]
                # Capping the row to 90 elems
                row = row[:max_length].copy()
                usable_data.append((label, filename, row))
                if ex.shape[0] > 19:
                    if len(filename.split("-")) == 1:
                        filename = filename + "-1"
                    else:
                        filename = filename.split("-")[0] + "-" + str(
                            int(filename.split("-")[1]) + 1)
                    new_elems.append((label, filename, ex))
            # if the row has less than max_length frames, repeat the row until it reaches max_length frames
            elif n_row < max_length:
                n_row_aux = n_row
                row_aux = row.copy()
                row_aux = np.repeat(
                    row_aux, [2 for _ in range(n_row_aux)], axis=0)
                if row_aux.shape[0] > max_length:
                    row_aux = row_aux[:max_length].copy()
                    usable_data.append((label, filename, row_aux))
                else:
                    new_elems.append((label, filename, row_aux))
        del data
        # if there are new elements, repeat the process
        if len(new_elems) != 0:
            cleaned_data = self.data_cleaner(new_elems, max_length)
            for cdl, filename, cdr in cleaned_data:
                usable_data.append((cdl, filename, cdr))

        return usable_data

    def dataset_cleaner(self) -> None:
        """Clean and standardize the animation dataset.
        
        Processes all CSV files in the dataset directory, ensuring uniform sequence length.
        Saves cleaned data to './dataset/splitted-animations' directory.
        """
        list_data: list[tuple[str, str, np.ndarray]] = []

        print("Loading dataset")
        files = glob.glob(os.path.join(
            self.__dataset_path, "*.csv"))
        for file in tqdm(files):
            try:
                list_data.append((file.split("/")[-1].split("_")[0], file.split("/")[-1].split(".")[0], pd.read_csv(
                    file, delimiter=",", dtype=np.float32).to_numpy()))
            except Exception as e:
                print(f"Error reading {file}")
                print(e)
                continue
        max_length = 90
        cleaned_data: list[tuple[str, str, np.ndarray]
                           ] = self.__dataset_cleaner_aux(list_data, max_length)
        errors: int = 0
        for label, filename, data in tqdm(cleaned_data):
            if len(data) != 90:
                print(f"Error with {label} with size {data.shape[0]}")
                errors += 1
        print(f"Errors: {errors} out of {len(cleaned_data)}, size mean: {
              np.mean([data.shape[0] for label, filename, data in cleaned_data])}")

        os.makedirs("./dataset/splitted-animations", exist_ok=True)
        for label, filename, row in tqdm(cleaned_data):
            np.savetxt(os.path.join("./dataset/splitted-animations",
                       filename + ".csv"), row, delimiter=",")

    def load_dataset_as_dataframe(self) -> pd.DataFrame:
        """Load and preprocess dataset into a pandas DataFrame.

        :return: DataFrame containing standardized animation sequences with labels
        :rtype: pd.DataFrame
        """
        data = []
        files = glob.glob(os.path.join(
            self.__dataset_path, "*.csv"))
        for file in tqdm(files):
            data_aux = pd.read_csv(file, delimiter=",", dtype=np.float32)
            data_aux = data_aux.to_numpy()[0:90:(round(self.__interval*90)), :]
            data_aux = data_aux.reshape(data_aux.shape[0], -1)
            category = file.split("/")[-1].split("_")[0]
            data_aux = np.append(data_aux, category)
            for row in data_aux:
                data.append(data_aux)

        header_size = data[0].shape[0]
        steps = len(range(0, 90, round(self.__interval*90)))
        header = pd.read_csv(
            files[0], delimiter=",", dtype=np.float32).columns.tolist()
        print(f"{header_size}, {steps}")
        header_final = header
        header_final = [f"{h}_{i}" for h in header for i in range(steps)]
        header_final = header_final + ["label"]
        print(len(header_final))
        data = pd.DataFrame(data, columns=header_final)
        return data

    def load_dataset(self) -> list[tuple[str, np.ndarray]]:
        """Load raw dataset as list of (label, array) tuples.

        :return: List of animation sequences with their category labels
        :rtype: list[tuple[str, np.ndarray]]
        """
        data = []
        files = glob.glob(os.path.join(
            self.__dataset_path, "*.csv"))

        print("Loading dataset")
        for file in tqdm(files):
            data_aux = pd.read_csv(file, delimiter=",", dtype=np.float32)
            data_aux = data_aux.to_numpy()[0:90:(round(self.__interval*90)), :]
            category = file.split("/")[-1].split("_")[0]
            data.append((category, data_aux))
        return data