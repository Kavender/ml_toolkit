import csv


def read_csv_file2list(filename, delimiter = ','):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter =delimiter, quoting=csv.QUOTE_NONE)
    data = list(reader)
    return data
