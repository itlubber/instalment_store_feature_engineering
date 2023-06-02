import os


basedir = os.path.abspath(os.path.dirname(__file__))

# 日志存储路径
logger_file = os.path.join(basedir, "logs/run.log")


# 数据库链接配置, 与 impala.dbapi 创建链接时的参数一致
impala_connect_options = dict(
    sid="geexdb",
    port=1521,
    host="172.16.104.109",
    password="zhangluping#0210",
    username="zhangluping",
    pool_size=16,
)


# 邮件推送设置
email_options = {
    "username": "geexfinance.com\GEEX2349",
    "password": "zhang123.",
    "server": "mail.geexfinance.com",
    "primary_smtp_address": "zhangluping@geexfinance.com",
    "receivers": [
        "zhangluping@geexfinance.com",
        "wangxiaonan@geexfinance.com",
        # "zhoulifeng@geexfinance.com",
        "xuyue1@geexfinance.com",
        # "zhaosuaichao@geexfinance.com",
    ]
}