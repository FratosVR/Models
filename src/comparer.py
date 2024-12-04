import DataLoader


def main() -> None:
    print("Downloading dataset")
    dl = DataLoader.DataLoader()
    print(dl.dataset_loader(0).head())


if __name__ == "__main__":
    main()
