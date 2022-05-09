"Common DB operation settings as parent class."
import time
import pandas as pd
import sqlite3
import psycopg2
import pymysql
from neo4j import GraphDatabase
from sqlite3 import Error as SQLError
from mysql.connector import as MSQLError
from neo4j.exceptions import ServiceUnavailable
from psycopg2 import OperationalError as PostgreSQLError

from common.types import DBClass
from common.utils import delay_time


class DBResults(object):

    def __init__(self, results):
        self.results = results

    def __enter__(self):
        return self.results

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results.close()
        self.results = None

    def __del__(self):
        if self.results:
            self.results.close()
            self.results = None


class BaseDB:
    instance = None

    def __new__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = super().__new__(BaseDB)
            return self.instance
        return self.instance

    def __init__(self, *args, **kwargs):
        self.retry_limit = 3
        self._type = kwargs.get("db_type", "sql")
        self.host_name = self._check_db_param(kwargs.get("hostname", None), "hostname")
        self.user_name = self._check_db_param(kwargs.get("username", None), "username")
        self.pass_word = self._check_db_param(kwargs.get("password", None), "password")
        self.port = self._check_db_param(kwargs.get("port", None), "port")
        self.database = self._check_db_param(kwargs.get("database", None), "database")
        self.connect()

    def connect(self):
        retry_count = 0
        while retry_count <= self.retry_limit:
            try:
                if self._type == DBClass.SQL.value:
                    self.db_conn = sqlite3.connect()
                elif self._type == DBClass.MYSQL.value:
                    self.db_conn = pymysql.connect(host=self.host_name, user=self.user_name, passwd=self.pass_word, port=self.port)
                elif self._type == DBClass.REDSHIFT.value:
                    self.db_conn = psycopg2.connect(host=self.host_name, user=self.user_name, password=self.pass_word,
                                                    port=self.port, database=self.database)
                else:
                    raise ValueError(f'Connection type {self._type} is not supported yet!')
                self.cursor = self.db_conn.cursor()
            except (SQLError, MSQLError, PostgreSQLError) as e:
                logger.error(f"Error {e} during db connection for {self._type}")
                if retry_count < self.retry_limit:
                    logger.debug(f"Retry to connect")
            retry_count += 1
            delay_time(10)

    def _check_db_param(self, value, param_name):
        if value:
            return value
        raise ValueError(f"You must supply a {param_name} when creating a {self._type} connection")

    def query(self, query, *args, **kwargs):
        result = self.db_conn.execute(query, *args, **kwargs)
        self._log_query(query)
        return result

    def execute(self, query, *args, **kwargs) -> CursorResult:
        return self.query(query, *args, **kwargs)

    def with_query(self, query, *args, **kwargs) -> DBResults:
        res = self.query(query, *args, **kwargs)
        return DBResults(res)

    def read_dataframe(
        self, query, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None,
    ):
        return pd.read_sql(
            query, self.db_conn, index_col, coerce_float, params, parse_dates, columns, chunksize,
        )

    def __del__(self):
        self.cursor.close()
        self.db_conn.close()


class Neo4jDB(BaseDB):
    
    def __init__(self, *args, **kwargs):
        super().__int__(self, *args, **kwargs)
        
    @overrides
    def connect(self):
        retry_count = 0
        while retry_count <= self.retry_limit:
            try:
                self.db_conn = GraphDatabase.driver(self.host_name, auth=(self.user_name, self.pass_word))
            except ServiceUnavailable as e:
                logger.error(f"Error {e} during db connection for {self._type}")
                if retry_count < self.retry_limit:
                    logger.debug(f"Retry to connect")
            retry_count += 1
            delay_time(10)

    @overrides
    def query(self, query, parameters=None, db=None):
        assert self.db_conn is not None, "Driver not initialized!"
        session, response = None, None
        
        try: 
            session = self.db_conn.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
