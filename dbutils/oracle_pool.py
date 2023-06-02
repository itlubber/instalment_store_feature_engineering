# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/23 13:12
@Author  : itlubber
@Site    : itlubber.art
"""

import traceback
import pandas as pd
import cx_Oracle as Oracle
from contextlib import contextmanager


def reduce_memory_usage(df):
    if isinstance(df, pd.DataFrame):
        for col, col_type in df.dtypes.iteritems():
            if col_type in ["int32", "int64", "float64", "float32"]:
                df[col] = pd.to_numeric(df[col], downcast="integer" if "int" in str(col_type) else "float")

    return df


class OraclePool:
    """
    1) 这里封装了一些有关oracle连接池的功能;
    2) sid和service_name，程序会自动判断哪个有值，
        若两个都有值，则默认使用service_name；
    3) 关于config的设置，注意只有 port 的值的类型是 int，以下是config样例:
        config = {
            'username':     'itlubber.art',
            'password':     'itlubber.art',
            'host':         '192.168.158.1',
            'port':         1521,
            'sid':          'itlubber.art',
            'service_name': 'itlubber.art'
        }
    """

    def __init__(self, username, password, host, port, sid=None, service_name=None, pool_size=5):
        """
        sid 和 service_name至少存在一个, 若都存在，则默认使用service_name
        """
        self.oracle_pool = self.get_pool(username, password, host, port, sid=sid, service_name=service_name,
                                         pool_size=pool_size)

    @staticmethod
    def get_pool(username, password, host, port, sid=None, service_name=None, pool_size=5):
        """
        ---------------------------------------------
        以下设置，根据需要进行配置
        max                 最大连接数
        min                 初始化时，连接池中至少创建的空闲连接。0表示不创建
        increment           每次增加的连接数量
        pool_size           连接池大小，这里为了避免连接风暴造成的资源浪费，设置了 max = min = pool_size
        """
        dsn = None
        if service_name:
            dsn = Oracle.makedsn(host, port, service_name=service_name)
        elif sid:
            dsn = Oracle.makedsn(host, port, sid=sid)

        return Oracle.SessionPool(user=username, password=password, dsn=dsn, min=pool_size, max=pool_size, increment=0,
                                  encoding='UTF-8', threaded=True)

    @property
    @contextmanager
    def pool(self):
        _conn = None
        _cursor = None
        try:
            _conn = self.oracle_pool.acquire()
            _cursor = _conn.cursor()
            yield _cursor
        finally:
            _conn.commit()
            self.oracle_pool.release(_conn)

    def query(self, query):
        res = pd.DataFrame()
        try:
            _conn = self.oracle_pool.acquire()
            res = pd.read_sql_query(query, _conn)
            # res = reduce_memory_usage(res)
        except:
            print(traceback.format_exc())
        finally:
            _conn.commit()
            self.oracle_pool.release(_conn)
            return res

    def execute(self, sql: str, *args, **kwargs):
        """
        执行sql语句
        :param sql:     str     sql语句
        :param args:    list    sql语句参数列表
        :return:        conn, cursor
        """
        with self.pool as cursor:
            cursor.execute(sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        """
        批量执行。
        :param sql:     str     sql语句
        :param args:    list    sql语句参数
        :return:        tuple   fetch结果
        """
        with self.pool as cursor:
            cursor.executemany(sql, *args, **kwargs)

    def fetchone(self, sql, *args, **kwargs) -> tuple:
        """
        获取全部结果
        :param sql:     str     sql语句
        :param args:    list    sql语句参数
        :return:        tuple   fetch结果
        """
        with self.pool as cursor:
            cursor.execute(sql, *args, **kwargs)
            return cursor.fetchone()

    def fetchall(self, sql, *args, **kwargs):
        """
        获取全部结果
        :param sql:     str     sql语句
        :param args:    list    sql语句参数
        :return:        tuple   fetch结果
        """
        with self.pool as cursor:
            cursor.execute(sql, *args, **kwargs)
            return cursor.fetchall()

    def __del__(self):
        """
        关闭连接池。
        """
        self.oracle_pool.close()
