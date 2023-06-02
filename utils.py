import os
import re
import time
import math
import random
import joblib
import numpy as np
import pandas as pd
import cx_Oracle as cx
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import config
from itlubber_logger import get_logger
from dbutils.oracle_pool import OraclePool


logger = get_logger(filename=config.logger_file, stream=True)
db_connect_pool = OraclePool(**config.impala_connect_options)


def load_pickle(file):
    return joblib.load(file)


def save_pickle(obj, file):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
        
    return joblib.dump(obj, file)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class OracleQuery:

    def __init__(self, sql, *args, **kwargs):
        self.sql = sql

        if args or kwargs:
            self.sql = self.sql.format(*args, **kwargs)
    
    def query(self, *args, **kwargs):
        return db_connect_pool.query(self.sql.format(*args, **kwargs))

    def insert(self, *args, **kwargs):
        db_connect_pool.executemany(self.sql, *args, **kwargs)


def to_category(df):
    cols = df.select_dtypes(include='object').columns
    for col in cols:
        ratio = len(df[col].value_counts()) / len(df)
        if ratio < 0.05:
            df[col] = df[col].astype('category')
    return df


def timer(func):
    """
    function cost time
    """
    def func_wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        logger.info('function {0}() cost time {1} s'.format(func.__name__, time_spend))
        return result

    return func_wrapper


def missing_data(data):
    missing = data.isnull().sum()
    available = data.count()
    total = (missing + available)
    percent = (data.isnull().sum()/data.isnull().count()*100).round(4)
    return pd.concat([missing, available, total, percent], axis=1, keys=['Missing', 'Available', 'Total', 'Percent']).sort_values(['Missing'], ascending=False)


def date_add(start: str, day: int=None, month: int=None, year: int=None, format: str=None):
    if day:
        if format is None:
            format = "%Y-%m-%d"
        return (datetime.strptime(start, format) + timedelta(days=day)).strftime(format)
    if month:
        if format is None:
            format = "%Y-%m"
        return (datetime.strptime(start, format) + timedelta(month=month)).strftime(format)
    if year:
        if format is None:
            format = "%Y"
        curr = datetime.strptime(start, format)
        return curr.strftime(format)


def reduce_memory_usage(df, deep=True, verbose=False):
    numeric2reduce = ["int16", "int32", "int64", "float64", "float32"]
    start_mem = 0
    if verbose:
        start_mem = df.memory_usage(deep=deep).sum() / 1024**2

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name

        if verbose and best_type is not None and best_type != str(col_type):
            logger.info(f"column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = df.memory_usage(deep=deep).sum() / 1024**2
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        logger.info(f"memory usage decreased from {start_mem:.2f}MB to {end_mem:.2f}MB ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    
    return df


def mapping_df_types(df):
    dtypedict = {}
    for i, j in zip(df.columns, df.dtypes):
        if "object" in str(j) or "str" in str(j):
            dtypedict.update({i: cx.DB_TYPE_VARCHAR})
        if "float" in str(j):
            dtypedict.update({i: cx.NUMBER})
        if "int" in str(j):
            dtypedict.update({i: cx.NUMBER})
        if "datetime" in str(j):
            dtypedict.update({i: cx.TIMESTAMP})
    return dtypedict


def generate_oracle_create_query(df, table_name, if_drop=True):
    try:
        res = db_connect_pool.fetchone(f"SELECT count(*) FROM all_tables WHERE table_name = '{table_name}'")

        if res[0] > 0 and if_drop:
            db_connect_pool.execute(f"drop table {table_name}")

        if res[0] == 0:
            create_sql = f'CREATE TABLE {table_name} (\n'
            for column in df.columns:
                column_type = df[column].dtype.name
                if column_type in ['object', 'str']:
                    column_type = 'VARCHAR2(255)'
                elif column_type in ('int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16'):
                    column_type = 'NUMBER'
                elif column_type == 'datetime64[ns]':
                    column_type = 'DATE'
                create_sql += f'    {column} {column_type},\n'

            create_sql = create_sql[:-2] + '\n)'

            db_connect_pool.execute(create_sql)
    except:
        logger.info(f"在执行创建表 {table_name} 时可能存在异常，有可能导致后续存储数据失败")

def generate_oracle_insert_query(df, table_name, batch_size=256):
    data = df.copy().reset_index(drop=True).replace(np.nan, None)

    insert_sql = f"INSERT INTO {table_name} \n({', '.join(list(data.columns))}) \nVALUES (:{', :'.join(list(data.columns))})"

    for i in tqdm(range(0, len(data), batch_size)):
        db_connect_pool.executemany(insert_sql, data.iloc[i:i+batch_size].values.tolist())


def str2date(date_str, format_=None):
    """
    字符串转时间, 默认自动推断格式
    :param date_str: 时间字符串
    :param format_: 格式
    :return: 对应的时间类型, 输入非字符串则原值输出
    """
    if not isinstance(date_str, str):
        date_str = str(date_str)
    
    if format_:
        return datetime.strptime(date_str, format_)

    s = re.match(r'(\d{4})\D+(\d{1,2})\D+(\d{1,2})(?:\D+(\d{1,2}))?(?:\D+(\d{1,2}))?(?:\D+(\d{1,2}))?\D*$', date_str)
    if s:
        result = [int(i) for i in s.groups() if i]
        return datetime(*result)
    
    s = re.match(r'(\d{4})\D*(\d{2})\D*(\d{2})?\D*(\d{2})?\D*(\d{2})?\D*(\d{2})?\D*$', date_str)
    if s:
        result = [int(i) for i in s.groups() if i]
        if len(result) == 2:
            result.append(1)
        return datetime(*result)
    
    raise ValueError("inference failed, please specify format")


def date_diff(start_time, end_time, unit):
    """
    日期差值计算
    :param start_time 起始时间
    :param end_time 终止时间
    :param unit 单位
    """
    if isinstance(start_time, str):
        start_time = str2date(start_time)

    if isinstance(end_time, str):
        end_time = str2date(end_time)

    year_diff = end_time.year - start_time.year
    month_diff = end_time.month - start_time.month
    time_diff = (end_time - start_time)
    day_diff = time_diff.days
    seconds_diff = time_diff.seconds

    if unit == 'Y':
        return year_diff
    elif unit == 'M':
        return year_diff * 12 + month_diff
    elif unit == 'D':
        return day_diff
    elif unit == 'A':
        return math.ceil(day_diff / 365)
    elif unit == 'S':
        return seconds_diff
    else:
        raise ValueError("date_diff unit input value error")


def df_date_diff(row, start_time_col, end_time_col, unit='D'):
    return date_diff(str(row[start_time_col]), str(row[end_time_col]), unit)


def calculate_diff(start_time, end_time, operate):
    reject_list = ['--', '0000.00', '00000', '00000.01', '000000', '000000.01', '0']

    if pd.isnull(start_time) or pd.isnull(end_time) or (start_time in reject_list) or (end_time in reject_list):
        return np.nan

    if isinstance(start_time, str):
        start_time = str2date(start_time)
    else:
        if not isinstance(start_time, datetime.datetime):
            raise ValueError("start_time type error,now is {}; should be string or datetime".format(type(start_time)))
    
    if isinstance(end_time, str):
        end_time = str2date(end_time)
    else:
        if not isinstance(end_time, datetime.datetime):
            raise ValueError("end_time type error,now is {}; should be string or datetime".format(type(end_time)))
    
    return date_diff(start_time, end_time, operate)


def last_day_of_month(any_day):
    """
    传入一个日期（datetime），获取该日期的月末日期时间
    """
    if isinstance(any_day, str):
        any_day = str2date(any_day)
    
    next_month = any_day.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def percentage2float(percentage):
    """
    将带中文 "%" 的百分数转换为浮点数
    """
    if isinstance(percentage, (int, float)):
        return float(percentage)
    elif isinstance(percentage, str):
        try:
            return float(percentage)
        except:
            if len(percentage) > 0 and percentage.endswith("%"):
                return float(percentage[:-1])


def clear(clear_string):
    """
    仅保留中文字符
    """
    return re.sub("[^\u4e00-\u9fa5]+", "", clear_string)


def json2str(json_dict):
    numpy_date_type = (np.int, np.int16, np.int32, np.int64, np.float, np.float16, np.float32, np.float64)
    item_values_str = [(x, float(y)) if isinstance(y, numpy_date_type) else (x, y) for x, y in json_dict.items()]
    new_dict = dict(item_values_str)
    return new_dict
