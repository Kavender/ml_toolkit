from enum import Enum

class FileFormat(Enum):
	CSV = "csv"
	TXT = "txt"
	JSON = "json"
	JOBLIB = "joblib"
	PICKLE = "pickle"
	MODEL = "model"


class DataFormat(Enum)：
	DATAFRAME = "dataframe"
	DICT = "dictionary"
	NESTDICT = "nested_dictionary"


class DBClass(Enum):
    SQL = "sql"
    MYSQL = "mysql"
    DYNAMO_DB = "dynamodb"
    REDSHIFT = "redshift"
    GRAPH_DB = "neo4j"
