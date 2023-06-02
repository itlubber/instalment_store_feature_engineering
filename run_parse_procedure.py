import os
import traceback
from utils import logger
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

from send_emails import send_email


def tick(replace=False, cache=False):
    logger.info("/" * 60)
    logger.info("开始执行python解析商户变量中 ......")
    if replace:
        if cache:
            os.system("python main.py --max_workers 8 --replace --batch_size 512 --unlock")
        else:
            os.system("python main.py --max_workers 8 --replace --clear --batch_size 512 --unlock")
    else:
        os.system("python main.py --max_workers 8 --clear --batch_size 512")
    logger.info("/" * 60)


def send():
    logger.info("通过邮箱发送商户变量监控数据 ......")
    try:
        send_email()
    except:
        logger.error("邮件发送失败")
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    # 全局文件锁, 仅允许存在一个解析的程序
    if os.path.isfile("parse_lock"):
        os.remove("parse_lock")

    # tick(replace=True, cache=True)
    scheduler = BlockingScheduler()
    scheduler.add_job(tick, 'cron', hour=4)
    scheduler.add_job(send, 'cron', hour=20)
    logger.info("设置定时任务执行商户变量解析相关任务")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
