import csv
import pickle
import json


# TODO: create a generic CommonFIleReader
def read_csv_file2list(filename, delimiter = ','):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter =delimiter, quoting=csv.QUOTE_NONE)
    data = list(reader)
    return data


def load_from_pickle(file_path):
    with open(file_path, 'rb') as outfile:
        data = pickle.load(outfile)
    return data


def load_json(path_to_file: str):
    "Helper function to load a JSON file"
    with open(path_to_file) as file:
        db = json.load(file)
    return db


def load_processed_records(file_path):
    data = []
    try:
        with open(file_path, 'rb') as infile:
            while True:
                data.append(pickle.load(infile))
    except EOFError:
        pass
    return data
