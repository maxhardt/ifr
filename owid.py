# script for downloading and cleaning owid covid data

# imports
import wget
import pandas as pd
import yaml

URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"


def download():
    filename = wget.download(URL, out="data")
    print("\n")
    return filename


def filter_countries(df):

    # load list of countries from yaml
    with open("countries.yaml", "rb") as f:
        countries = yaml.load(f)

    df = df.loc[df["iso_code"].isin(countries["iso_code"])]

    return df


def preprocess(filepath):

    df = pd.read_csv(filepath, sep=",", header="infer", index_col=False)
    df["date"] = pd.to_datetime(df["date"])

    columns = [
        "iso_code",
        "location",
        "date",
        "population",
        "total_tests",
        "total_cases",
        "total_deaths",
        "hospital_beds_per_thousand",
        "human_development_index",
        "life_expectancy"
    ]

    df = df[columns]

    # retain selected countries
    df = filter_countries(df)

    # drop all rows where no test data is available
    df = df.dropna(axis=0, how="all", subset=["total_tests"])

    # keep only the most recent row per location
    df = df.sort_values("date").drop_duplicates("iso_code", keep='last')

    return df[columns]


if __name__ == "__main__":
    filename = download()
    df = preprocess(filename)
    df.to_csv("./data/data_processed.csv", index=False)
