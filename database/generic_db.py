"Common DB operation settings as parent class."

class GenericDB:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, sql, *args, **kwargs):
        raise NotImplementedError

    def query_many(self, sql, *args, **kwargs):
        raise NotImplementedError

    def get_item(self, *args, **kwargs):
        raise NotImplementedError

    def put_item(self, *args, **kwargs):
        raise NotImplementedError

    def delete_item(self, *args, **kwargs):
        raise NotImplementedError

    def close_connection(self, *args, **kwargs):
        raise NotImplementedError