import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import cx_Oracle as cx
from datetime import datetime
from exchangelib import Credentials, Account, DELEGATE, Configuration, NTLM, Message, Mailbox, HTMLBody, FileAttachment

from config import email_options


def send_email():
    today = datetime.now().strftime("%Y-%m-%d")
    config = Configuration(credentials=Credentials(email_options["username"], email_options["password"]),
                           server=email_options["server"], auth_type="NTLM")
    account = Account(email_options["primary_smtp_address"], config=config, autodiscover=False, access_type=DELEGATE)

    item = Message(
        account=account,
        folder=account.sent,
        subject=f'XXXX-商户收单变量表-{today}',
        body=HTMLBody(f"邮件主题: 商户监控相关变量<br/>更新时间: {today}<br/>附件内容: 商户收单变量数据字典、商户衍生变量数据字典、商户变量表"),
        to_recipients=[Mailbox(email_address=e) for e in email_options["receivers"]],
    )

    item.attach(FileAttachment(name=f'XXXX-商户收单变量表-{today}.xlsx', content=open('商户相关变量.xlsx', 'rb').read()))

    item.send()


if __name__ == '__main__':
    send_email()
