import sys
import traceback
import atexit
import warnings

warnings.filterwarnings("ignore")

import os
import fuckit
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil import parser, rrule
from geopy.distance import geodesic
from utils import logger, seed_everything
from dateutil.relativedelta import relativedelta

from utils import *
from tools.excel_writer import ExcelWriter
from feature_calculator.oracle_query import *


def deal_store_zjj_order_features(data, today, store_code):
    data = data.copy()

    if isinstance(today, str):
        today = parser.parse(today)

    results = {}

    def deal_last_nopay_days(data, start, end, key="N_PAY_AMOUNT", aggfun="sum"):
        """
        最近一段时间内的无结算天数
        """
        if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
            cci_order = data["cci_order"].copy()
            cci_order = cci_order[cci_order["D_CREATE_TIME"] > start]
            if len(cci_order) > 0:
                return (end - start).days - (
                        cci_order.set_index("D_CREATE_TIME").resample("D")[key].agg(eval(aggfun)) > 0).sum()
            else:
                return (end - start).days
        else:
            return -1

    # 近一个月无结算天数
    results["GEXshvar070"] = deal_last_nopay_days(data, today - relativedelta(months=1), today, key="C_ORDER_NO",
                                                  aggfun="len")
    # 近3个月无结算天数
    results["GEXshvar101"] = deal_last_nopay_days(data, today - relativedelta(months=3), today, key="C_ORDER_NO",
                                                  aggfun="len")
    # 近7天流入为0的天数
    results["GEXshvar061_1"] = deal_last_nopay_days(data, today - relativedelta(days=7), today, key="N_PAY_AMOUNT",
                                                    aggfun="sum")
    # 近14天流入为0的天数
    results["GEXshvar025_1"] = deal_last_nopay_days(data, today - relativedelta(days=14), today, key="N_PAY_AMOUNT",
                                                    aggfun="sum")
    # 近1个月流入为0的天数
    results["GEXshvar029_1"] = deal_last_nopay_days(data, today - relativedelta(months=1), today, key="N_PAY_AMOUNT",
                                                    aggfun="sum")
    # 近3个月流入为0的天数
    results["GEXshvar043_1"] = deal_last_nopay_days(data, today - relativedelta(months=3), today, key="N_PAY_AMOUNT",
                                                    aggfun="sum")

    def deal_last_nopay_days(data, start, end=today):
        """
        最近一段时间的放款额（万元）
        """
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            curr_loan_base = loan_base[(loan_base["D_LOAN_DATE"] > start) & (loan_base["D_LOAN_DATE"] <= end)]
            if len(curr_loan_base) > 0:
                return curr_loan_base["N_LOAN_AMOUNT"].sum() / 10000
            else:
                return 0
        else:
            return -1

    # 近7天的放款额（万元）
    results["GEXshvar061_2"] = deal_last_nopay_days(data, today - relativedelta(days=7), end=today)
    # 近14天的放款额（万元）
    results["GEXshvar025_2"] = deal_last_nopay_days(data, today - relativedelta(days=14), end=today)
    # 近1个月的放款额（万元）
    results["GEXshvar029_2"] = deal_last_nopay_days(data, today - relativedelta(months=1), end=today)
    # 近3个月的放款额（万元）
    results["GEXshvar043_2"] = deal_last_nopay_days(data, today - relativedelta(months=3), end=today)

    # 最近n个月流入额 / 最近n个月放款额
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0 and data.get("loan_base") is None and len(
            data["loan_base"]) > 0:
        cci_order = data["cci_order"].copy()
        loan_base = data["loan_base"].copy()

        # 当前实际收单比（月均结算额/月均放款额）
        # 月均结算额
        N_PAY_AMOUNT_AVG_MONTHLY_CCI = cci_order.set_index("D_CREATE_TIME").resample("M")["N_PAY_AMOUNT"].sum().mean()
        # 月均放款额
        N_LOAN_AMOUNT_AVG_MONTHLY = loan_base.set_index("D_LOAN_DATE").resample("M")["N_LOAN_AMOUNT"].sum().mean()
        if N_LOAN_AMOUNT_AVG_MONTHLY > 0:
            results["GEXshvar012"] = N_PAY_AMOUNT_AVG_MONTHLY_CCI / N_LOAN_AMOUNT_AVG_MONTHLY
        else:
            results["GEXshvar012"] = -1

        # 近1个月的流入额/近1个月放款额
        curr_loan_base = loan_base[loan_base["D_LOAN_DATE"] > today - relativedelta(months=1)]
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=1)]
        if len(curr_loan_base) != 0 or len(curr_cci_order) != 0:
            results["GEXshvar026"] = curr_cci_order["N_PAY_AMOUNT"].sum() / curr_loan_base["N_LOAN_AMOUNT"].sum()
        else:
            results["GEXshvar026"] = -1

        # 近3个月的流入额/近3个月放款额
        curr_loan_base = loan_base[loan_base["D_LOAN_DATE"] > today - relativedelta(months=3)]
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=3)]
        if len(curr_loan_base) != 0 or len(curr_cci_order) != 0:
            results["GEXshvar033"] = curr_cci_order["N_PAY_AMOUNT"].sum() / curr_loan_base["N_LOAN_AMOUNT"].sum()
        else:
            results["GEXshvar033"] = -1

        # 近7天的流入额/近7天放款额
        curr_loan_base = loan_base[loan_base["D_LOAN_DATE"] > today - relativedelta(days=7)]
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(days=7)]
        if len(curr_loan_base) != 0 or len(curr_cci_order) != 0:
            results["GEXshvar059"] = curr_cci_order["N_PAY_AMOUNT"].sum() / curr_loan_base["N_LOAN_AMOUNT"].sum()
        else:
            results["GEXshvar059"] = -1

    else:
        results["GEXshvar026"] = -1
        results["GEXshvar033"] = -1
        results["GEXshvar059"] = -1

    # 净流入相关变量
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()
        cci_refund = data["cci_refund"].copy()

        if cci_refund is not None and len(cci_refund) > 0:
            cci_order = cci_order.merge(cci_refund[["C_ORDER_NO", "N_REFUND_AMOUNT"]], on="C_ORDER_NO", how="left")
            cci_order["N_PAY_AMOUNT"] = cci_order["N_PAY_AMOUNT"] - cci_order["N_REFUND_AMOUNT"].fillna(0)

        # ////////////////////////////////////// 当月净流入相关变量 ////////////////////////////////////// #
        # 日均净流入
        store_daily_pay_amount_avg = cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum().mean()

        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] == today.strftime("%Y-%m")]
        yoy_cci_order = cci_order[cci_order["D_CREATE_TIME"] == (today - relativedelta(years=1)).strftime("%Y-%m")]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 当月日均资金净流入
            curr_daily_pay_amount_avg = cur_cci_order.set_index("D_CREATE_TIME").resample("D")[
                "N_PAY_AMOUNT"].sum().mean()
            # 当月日均资金净流入/日均净流入
            results["GEXshvar013"] = curr_daily_pay_amount_avg / store_daily_pay_amount_avg
            # 当月资金净流入
            results["GEXshvar014"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 当月资金净流入（去除万元整数交易）
            results["GEXshvar015"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar013"] = 0
            results["GEXshvar014"] = 0
            results["GEXshvar015"] = 0

        # 当月资金净流入/去年同期净流入
        if yoy_cci_order is not None and len(yoy_cci_order) > 0 and cur_cci_order is not None and len(
                cur_cci_order) > 0:
            results["GEXshvar016"] = cur_cci_order["N_PAY_AMOUNT"].sum() / yoy_cci_order["N_PAY_AMOUNT"].sum()
        else:
            results["GEXshvar016"] = np.nan

        # ////////////////////////////////////// 近14天净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(days=14)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        yoy_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(years=1, days=14)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(years=1)).strftime("%Y-%m-%d"))]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近14日日均资金净流入
            curr_daily_pay_amount_avg = cur_cci_order.set_index("D_CREATE_TIME").resample("D")[
                "N_PAY_AMOUNT"].sum().mean()
            # 近14日日均资金净流入/日均净流入
            results["GEXshvar022"] = curr_daily_pay_amount_avg / store_daily_pay_amount_avg
            # 近14日资金净流入
            results["GEXshvar023"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 近14日资金净流入（去除万元整数交易）
            results["GEXshvar023_1"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar022"] = 0
            results["GEXshvar023"] = 0
            results["GEXshvar023_1"] = 0

        # 近14日日均资金净流入/去年同期近14日日均资金净流入
        if yoy_cci_order is not None and len(yoy_cci_order) > 0 and cur_cci_order is not None and len(
                cur_cci_order) > 0:
            results["GEXshvar021"] = cur_cci_order["N_PAY_AMOUNT"].sum() / yoy_cci_order["N_PAY_AMOUNT"].sum()
        else:
            results["GEXshvar021"] = np.nan

        # ////////////////////////////////////// 近3天净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(days=3)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近3日日均资金净流入
            curr_daily_pay_amount_avg = cur_cci_order.set_index("D_CREATE_TIME").resample("D")[
                "N_PAY_AMOUNT"].sum().mean()
            # 近3日日均资金净流入/日均净流入
            results["GEXshvar062"] = curr_daily_pay_amount_avg / store_daily_pay_amount_avg
            # 近3日资金净流入
            results["GEXshvar063"] = cur_cci_order["N_PAY_AMOUNT"].sum()
        else:
            results["GEXshvar062"] = 0
            results["GEXshvar063"] = 0

        # ////////////////////////////////////// 近3个月净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=3)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        yoy_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(years=1, months=3)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(years=1)).strftime("%Y-%m-%d"))]
        mom_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=6)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(months=3)).strftime("%Y-%m-%d"))]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近3个月日均资金净流入
            curr_daily_pay_amount_avg = cur_cci_order.set_index("D_CREATE_TIME").resample("D")[
                "N_PAY_AMOUNT"].sum().mean()
            # 近3个月日均资金净流入/日均净流入
            results["GEXshvar102_1"] = curr_daily_pay_amount_avg / (store_daily_pay_amount_avg + 1e-6)
            # 近3个月资金净流入
            results["GEXshvar102"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 近3个月资金净流入（去除万元整数交易）
            results["GEXshvar103"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar102_1"] = 0
            results["GEXshvar102"] = 0
            results["GEXshvar103"] = 0

        # 近3个月日均资金净流入/去年同期近3个月日均资金净流入
        if yoy_cci_order is not None and len(yoy_cci_order) > 0 and cur_cci_order is not None and len(
                cur_cci_order) > 0:
            results["GEXshvar104"] = cur_cci_order["N_PAY_AMOUNT"].sum() / (yoy_cci_order["N_PAY_AMOUNT"].sum() + 1e-6)
        else:
            results["GEXshvar104"] = np.nan

        # 最近3个月资金净流入的环比增长率
        if cur_cci_order is not None and len(cur_cci_order) > 0 and mom_cci_order is not None and len(
                mom_cci_order) > 0:
            curr = cur_cci_order["N_PAY_AMOUNT"].sum()
            base = mom_cci_order["N_PAY_AMOUNT"].sum()
            results["GEXshvar105"] = (curr - base) / base
        else:
            results["GEXshvar105"] = np.nan

        # ////////////////////////////////////// 近6个月净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=6)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        yoy_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(years=1, months=6)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(years=1)).strftime("%Y-%m-%d"))]
        mom_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=12)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(months=3)).strftime("%Y-%m-%d"))]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近6个月资金净流入
            results["GEXshvar124"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 近6个月资金净流入（去除万元整数交易）
            results["GEXshvar125"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar124"] = 0
            results["GEXshvar125"] = 0

        # 近6个月日均资金净流入/去年同期近6个月日均资金净流入
        if yoy_cci_order is not None and len(yoy_cci_order) > 0 and cur_cci_order is not None and len(
                cur_cci_order) > 0:
            results["GEXshvar126"] = cur_cci_order["N_PAY_AMOUNT"].sum() / yoy_cci_order["N_PAY_AMOUNT"].sum()
        else:
            results["GEXshvar126"] = np.nan

        # 最近6个月资金净流入的环比增长率
        if cur_cci_order is not None and len(cur_cci_order) > 0 and mom_cci_order is not None and len(
                mom_cci_order) > 0:
            curr = cur_cci_order["N_PAY_AMOUNT"].sum()
            base = mom_cci_order["N_PAY_AMOUNT"].sum()
            results["GEXshvar127"] = (curr - base) / base
        else:
            results["GEXshvar127"] = np.nan

        # ////////////////////////////////////// 近12个月净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=12)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        yoy_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(years=1, months=24)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(years=1)).strftime("%Y-%m-%d"))]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近12个月资金净流入
            results["GEXshvar086"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 近12个月资金净流入（去除万元整数交易）
            results["GEXshvar087"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar086"] = 0
            results["GEXshvar087"] = 0

        # 近12个月日均资金净流入/去年同期近12个月日均资金净流入
        if yoy_cci_order is not None and len(yoy_cci_order) > 0 and cur_cci_order is not None and len(
                cur_cci_order) > 0:
            results["GEXshvar088"] = cur_cci_order["N_PAY_AMOUNT"].sum() / yoy_cci_order["N_PAY_AMOUNT"].sum()
        else:
            results["GEXshvar088"] = np.nan

        # ////////////////////////////////////// 近24个月净流入相关变量 ////////////////////////////////////// #
        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] > (today - relativedelta(months=24)).strftime("%Y-%m-%d")]

        if cur_cci_order is not None and len(cur_cci_order) > 0:
            # 近24个月资金净流入
            results["GEXshvar086"] = cur_cci_order["N_PAY_AMOUNT"].sum()
            # 近24个月资金净流入（去除万元整数交易）
            results["GEXshvar087"] = cur_cci_order["N_PAY_AMOUNT"].apply(lambda x: 0 if x % 10000 == 0 else x).sum()
        else:
            results["GEXshvar089"] = 0
            results["GEXshvar090"] = 0

    else:
        results["GEXshvar013"] = -1
        results["GEXshvar014"] = -1
        results["GEXshvar015"] = -1
        results["GEXshvar016"] = np.nan
        results["GEXshvar022"] = -1
        results["GEXshvar023"] = -1
        results["GEXshvar023_1"] = -1
        results["GEXshvar021"] = np.nan
        results["GEXshvar062"] = -1
        results["GEXshvar063"] = -1
        results["GEXshvar102_1"] = -1
        results["GEXshvar102"] = -1
        results["GEXshvar103"] = -1
        results["GEXshvar104"] = np.nan
        results["GEXshvar105"] = np.nan
        results["GEXshvar124"] = -1
        results["GEXshvar125"] = -1
        results["GEXshvar126"] = np.nan
        results["GEXshvar127"] = np.nan
        results["GEXshvar086"] = -1
        results["GEXshvar087"] = -1
        results["GEXshvar088"] = np.nan
        results["GEXshvar089"] = -1
        results["GEXshvar090"] = -1

    # 最近n个月流入笔数相关的问题
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()

        # 最近12个月流入笔数
        results["GEXshvar020_1"] = len(cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=12)])
        # 最近6个月流入笔数
        results["GEXshvar055_1"] = len(cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=6)])
        # 最近3个月流入笔数
        results["GEXshvar046_1"] = len(cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=3)])
        # 最近1个月流入笔数
        results["GEXshvar032_1"] = len(cci_order[cci_order["D_CREATE_TIME"] > today - relativedelta(months=1)])
    else:
        results["GEXshvar020_1"] = -1
        results["GEXshvar055_1"] = -1
        results["GEXshvar046_1"] = -1
        results["GEXshvar032_1"] = -1

    # 近n个月小于1000元的天数
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()

        # 近1个月日流入小于1000元的天数
        curr_date = today - relativedelta(months=1)
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > curr_date]
        if len(curr_cci_order) > 0:
            over_threshold_days = (
                    curr_cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum() >= 1000).sum()
            results["GEXshvar030"] = (today - curr_date).days - over_threshold_days
        else:
            results["GEXshvar030"] = (today - curr_date).days

        # 近3个月日流入小于1000元的天数
        curr_date = today - relativedelta(months=3)
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > curr_date]
        if len(curr_cci_order) > 0:
            over_threshold_days = (
                    curr_cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum() >= 1000).sum()
            results["GEXshvar044"] = (today - curr_date).days - over_threshold_days
        else:
            results["GEXshvar044"] = (today - curr_date).days

        # 近6个月日流入小于1000元的天数
        curr_date = today - relativedelta(months=6)
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > curr_date]
        if len(curr_cci_order) > 0:
            over_threshold_days = (
                    curr_cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum() >= 1000).sum()
            results["GEXshvar052"] = (today - curr_date).days - over_threshold_days
        else:
            results["GEXshvar052"] = (today - curr_date).days

        # 近12个月日流入小于1000元的天数
        curr_date = today - relativedelta(months=12)
        curr_cci_order = cci_order[cci_order["D_CREATE_TIME"] > curr_date]
        if len(curr_cci_order) > 0:
            over_threshold_days = (
                    curr_cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum() >= 1000).sum()
            results["GEXshvar018"] = (today - curr_date).days - over_threshold_days
        else:
            results["GEXshvar018"] = (today - curr_date).days
    else:
        results["GEXshvar030"] = -1
        results["GEXshvar044"] = -1
        results["GEXshvar052"] = -1
        results["GEXshvar018"] = -1

    # 近n个月退货次数
    if data.get("cci_refund") is not None and len(data["cci_refund"]) > 0:
        cci_refund = data["cci_refund"].copy()
        cci_refund = cci_refund[cci_refund["D_UPDATE_TIME"] <= today]
        # 近12个月退货次数
        results["GEXshvar019"] = len(cci_refund[cci_refund["D_CREATE_TIME"] > today - relativedelta(months=12)])
        # 近6个月退货次数
        results["GEXshvar053"] = len(cci_refund[cci_refund["D_CREATE_TIME"] > today - relativedelta(months=6)])
        # 近3个月退货次数
        results["GEXshvar045"] = len(cci_refund[cci_refund["D_CREATE_TIME"] > today - relativedelta(months=3)])
        # 近1个月退货次数
        results["GEXshvar031"] = len(cci_refund[cci_refund["D_CREATE_TIME"] > today - relativedelta(months=1)])
    else:
        results["GEXshvar019"] = -1
        results["GEXshvar053"] = -1
        results["GEXshvar045"] = -1
        results["GEXshvar031"] = -1

    # 日均交易额相关变量加工
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()
        cci_refund = data["cci_refund"].copy()

        if cci_refund is not None and len(cci_refund) > 0:
            cci_order = cci_order.merge(cci_refund[["C_ORDER_NO", "N_REFUND_AMOUNT"]], on="C_ORDER_NO", how="left")
            # cci_order["N_PAY_AMOUNT"] = cci_order["N_PAY_AMOUNT"] - cci_order["N_REFUND_AMOUNT"].fillna(0)

        # 交易额年日均
        obs_date = today - relativedelta(years=1)
        obs_cci_order = cci_order[cci_order["D_CREATE_TIME"] > obs_date]
        results["GEXshvar004"] = obs_cci_order["N_PAY_AMOUNT"].sum() / (today - obs_date).days

        # 交易额季日均
        obs_date = today - relativedelta(months=3)
        obs_cci_order = cci_order[cci_order["D_CREATE_TIME"] > obs_date]
        results["GEXshvar003"] = obs_cci_order["N_PAY_AMOUNT"].sum() / (today - obs_date).days

        # 交易额月日均
        obs_date = today - relativedelta(months=1)
        obs_cci_order = cci_order[cci_order["D_CREATE_TIME"] > obs_date]
        results["GEXshvar006"] = obs_cci_order["N_PAY_AMOUNT"].sum() / (today - obs_date).days

        # 存款月均余额变化率（金融资产变化率）
        obs_date = today - relativedelta(months=1)
        obs_cci_order = cci_order[cci_order["D_CREATE_TIME"] <= obs_date]

        if obs_cci_order is not None and len(obs_cci_order) > 0:
            cur_amount_monthly_avg = cci_order.set_index("D_CREATE_TIME").resample("M")["N_PAY_AMOUNT"].sum().mean()
            obs_amount_monthly_avg = obs_cci_order.set_index("D_CREATE_TIME").resample("M")["N_PAY_AMOUNT"].sum().mean()
            results["GEXshvar005"] = (cur_amount_monthly_avg - obs_amount_monthly_avg) / obs_amount_monthly_avg
        else:
            results["GEXshvar005"] = np.nan

    else:
        results["GEXshvar003"] = -1
        results["GEXshvar004"] = -1
        results["GEXshvar005"] = np.nan
        results["GEXshvar006"] = -1

    # 近n个月环比结算金额 or 笔数增长率
    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()

        # ////////////////////////////////////// 近1个月环比结算增长率 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=1)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        mom_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=2)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(months=1)).strftime("%Y-%m-%d"))]

        # 最近1个月结算的环比增长率
        if cur_cci_order is not None and len(cur_cci_order) > 0 and mom_cci_order is not None and len(
                mom_cci_order) > 0:
            curr_amount = cur_cci_order["N_PAY_AMOUNT"].sum()
            base_amount = mom_cci_order["N_PAY_AMOUNT"].sum()
            curr_count = cur_cci_order["C_ORDER_NO"].nunique()
            base_count = mom_cci_order["C_ORDER_NO"].nunique()

            results["GEXshvar066"] = (curr_amount - base_amount) / base_amount
            results["GEXshvar065"] = (curr_count - base_count) / base_count
        else:
            results["GEXshvar066"] = np.nan
            results["GEXshvar065"] = np.nan

        # ////////////////////////////////////// 近1周环比结算增长率 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        mom_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(weeks=2)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(weeks=1)).strftime("%Y-%m-%d"))]

        # 最近1周结算的环比增长率
        if cur_cci_order is not None and len(cur_cci_order) > 0 and mom_cci_order is not None and len(
                mom_cci_order) > 0:
            curr_amount = cur_cci_order["N_PAY_AMOUNT"].sum()
            base_amount = mom_cci_order["N_PAY_AMOUNT"].sum()
            curr_count = cur_cci_order["C_ORDER_NO"].nunique()
            base_count = mom_cci_order["C_ORDER_NO"].nunique()

            results["GEXshvar072"] = (curr_amount - base_amount) / base_amount
            results["GEXshvar071"] = (curr_count - base_count) / base_count
        else:
            results["GEXshvar072"] = np.nan
            results["GEXshvar071"] = np.nan

        # ////////////////////////////////////// 近3个月环比结算增长率 ////////////////////////////////////// #
        cur_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=3)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= today.strftime("%Y-%m-%d"))]
        mom_cci_order = cci_order[
            (cci_order["D_CREATE_TIME"] > (today - relativedelta(months=6)).strftime("%Y-%m-%d")) & (
                    cci_order["D_CREATE_TIME"] <= (today - relativedelta(months=3)).strftime("%Y-%m-%d"))]

        # 最近3个月结算的环比增长率
        if cur_cci_order is not None and len(cur_cci_order) > 0 and mom_cci_order is not None and len(
                mom_cci_order) > 0:
            curr_amount = cur_cci_order["N_PAY_AMOUNT"].sum()
            base_amount = mom_cci_order["N_PAY_AMOUNT"].sum()
            curr_count = cur_cci_order["C_ORDER_NO"].nunique()
            base_count = mom_cci_order["C_ORDER_NO"].nunique()

            results["GEXshvar036"] = (curr_amount - base_amount) / base_amount
            results["GEXshvar035"] = (curr_count - base_count) / base_count
        else:
            results["GEXshvar036"] = np.nan
            results["GEXshvar035"] = np.nan

    else:
        results["GEXshvar066"] = np.nan
        results["GEXshvar065"] = np.nan
        results["GEXshvar072"] = np.nan
        results["GEXshvar071"] = np.nan
        results["GEXshvar036"] = np.nan
        results["GEXshvar035"] = np.nan

    if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
        cci_order = data["cci_order"].copy()

        # 近1个月交易的单笔金额的方差
        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] >= (today - relativedelta(months=1)).strftime("%Y-%m-%d")]
        if cur_cci_order is not None and len(cur_cci_order) > 0:
            results["GEXshvar027"] = np.nanvar(cur_cci_order["N_PAY_AMOUNT"].values)
        else:
            results["GEXshvar027"] = -1

        # 近3个月交易的单笔金额的方差
        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] >= (today - relativedelta(months=3)).strftime("%Y-%m-%d")]
        if cur_cci_order is not None and len(cur_cci_order) > 0:
            results["GEXshvar038"] = np.nanvar(cur_cci_order["N_PAY_AMOUNT"].values)
        else:
            results["GEXshvar038"] = -1

        # 近3天交易的单笔金额的方差
        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] >= (today - relativedelta(days=3)).strftime("%Y-%m-%d")]
        if cur_cci_order is not None and len(cur_cci_order) > 0:
            results["GEXshvar048"] = np.nanvar(cur_cci_order["N_PAY_AMOUNT"].values)
        else:
            results["GEXshvar048"] = -1

        # 近3天交易的单笔金额的方差
        cur_cci_order = cci_order[cci_order["D_CREATE_TIME"] >= (today - relativedelta(days=7)).strftime("%Y-%m-%d")]
        if cur_cci_order is not None and len(cur_cci_order) > 0:
            results["GEXshvar060"] = np.nanvar(cur_cci_order["N_PAY_AMOUNT"].values)
        else:
            results["GEXshvar060"] = -1
    else:
        results["GEXshvar027"] = -1
        results["GEXshvar038"] = -1
        results["GEXshvar048"] = -1

    def deal_continued_overdue_days(data, month=3, day=3, aggfun="max"):
        """
        最近n个月连续m日最大流入额
        """
        if data.get("cci_order") is not None and len(data["cci_order"]) > 0:
            cci_order = data["cci_order"].copy()
            cur_cci_order = cci_order[
                cci_order["D_CREATE_TIME"] > (today - relativedelta(months=month)).strftime("%Y-%m-%d")]

            if len(cur_cci_order) > 0:
                cur_cci_order_daily = cur_cci_order.set_index("D_CREATE_TIME").resample("D")["N_PAY_AMOUNT"].sum()
                return cur_cci_order_daily.rolling(window=day, min_periods=1).sum().agg(aggfun)
            else:
                return 0
        else:
            return -1

    results["GEXshvar128"] = deal_continued_overdue_days(data, month=1, day=1, aggfun="max")
    results["GEXshvar130"] = deal_continued_overdue_days(data, month=1, day=3, aggfun="max")
    results["GEXshvar132"] = deal_continued_overdue_days(data, month=1, day=3, aggfun="min")
    results["GEXshvar092"] = deal_continued_overdue_days(data, month=3, day=3, aggfun="max")
    results["GEXshvar093"] = deal_continued_overdue_days(data, month=3, day=3, aggfun="min")
    results["GEXshvar095"] = deal_continued_overdue_days(data, month=3, day=7, aggfun="max")
    results["GEXshvar098"] = deal_continued_overdue_days(data, month=3, day=7, aggfun="min")
    results["GEXshvar112"] = deal_continued_overdue_days(data, month=6, day=3, aggfun="max")
    results["GEXshvar115"] = deal_continued_overdue_days(data, month=6, day=3, aggfun="min")
    results["GEXshvar118"] = deal_continued_overdue_days(data, month=6, day=7, aggfun="max")
    results["GEXshvar121"] = deal_continued_overdue_days(data, month=6, day=7, aggfun="min")
    results["GEXshvar106"] = deal_continued_overdue_days(data, month=6, day=14, aggfun="max")
    results["GEXshvar109"] = deal_continued_overdue_days(data, month=6, day=14, aggfun="min")

    # 交互变量
    # 近1个月退货次数/近1个月流入笔数
    results["GEXshvar032"] = -1 if results["GEXshvar031"] == -1 or results["GEXshvar032_1"] == -1 else (
        results["GEXshvar031"] / results["GEXshvar032_1"] if results["GEXshvar032_1"] > 0 else 0)
    # 近3个月退货次数/近3个月流入笔数
    results["GEXshvar046"] = -1 if results["GEXshvar045"] == -1 or results["GEXshvar046_1"] == -1 else (
        results["GEXshvar045"] / results["GEXshvar046_1"] if results["GEXshvar046_1"] > 0 else 0)
    # 近6个月退货次数/近6个月流入笔数
    results["GEXshvar055"] = -1 if results["GEXshvar053"] == -1 or results["GEXshvar055_1"] == -1 else (
        results["GEXshvar053"] / results["GEXshvar055_1"] if results["GEXshvar055_1"] > 0 else 0)
    # 近12个月退货次数/近12个月流入笔数
    results["GEXshvar020"] = -1 if results["GEXshvar019"] == -1 or results["GEXshvar020_1"] == -1 else (
        results["GEXshvar019"] / results["GEXshvar020_1"] if results["GEXshvar020_1"] > 0 else 0)
    # 近6个月退货次数 / 近12个月流入笔数
    results["GEXshvar054_1"] = -1 if results["GEXshvar053"] == -1 or results["GEXshvar020_1"] == -1 else (
        results["GEXshvar053"] / results["GEXshvar020_1"] if results["GEXshvar020_1"] > 0 else 0)

    # 最近一个月单日最大流入额 / 年日均流入额
    results["GEXshvar129"] = results["GEXshvar128"] / results["GEXshvar004"] if results["GEXshvar004"] > 0 else -1
    # 最近一个月连续3日最大流入额/（3*年日均流入额）
    results["GEXshvar131"] = results["GEXshvar130"] / (results["GEXshvar004"] * 3) if results["GEXshvar004"] > 0 else -1
    # 最近一个月连续3日最小流入额/（3*年日均流入额）
    results["GEXshvar133"] = results["GEXshvar132"] / (results["GEXshvar004"] * 3) if results["GEXshvar004"] > 0 else -1
    # 最近3个月连续3日最小流入额/（3*年日均流入额）
    results["GEXshvar094"] = results["GEXshvar093"] / (results["GEXshvar004"] * 3) if results["GEXshvar004"] > 0 else -1
    # 最近3个月连续7日最大流入额/（7*年日均流入额）
    results["GEXshvar096"] = results["GEXshvar095"] / (results["GEXshvar004"] * 7) if results["GEXshvar004"] > 0 else -1
    # 最近3个月连续7日最小流入额/（7*年日均流入额）
    results["GEXshvar099"] = results["GEXshvar098"] / (results["GEXshvar004"] * 7) if results["GEXshvar004"] > 0 else -1
    # 最近6个月连续14日最大流入额/（14*年日均流入额）
    results["GEXshvar107"] = results["GEXshvar106"] / (results["GEXshvar004"] * 14) if results[
                                                                                           "GEXshvar004"] > 0 else -1
    # 最近6个月连续14日最小流入额/（14*年日均流入额）
    results["GEXshvar110"] = results["GEXshvar109"] / (results["GEXshvar004"] * 14) if results[
                                                                                           "GEXshvar004"] > 0 else -1
    # 最近6个月连续3日最大流入额/（3*年日均流入额）
    results["GEXshvar113"] = results["GEXshvar112"] / (results["GEXshvar004"] * 3) if results["GEXshvar004"] > 0 else -1
    # 最近6个月连续3日最小流入额/（3*年日均流入额）
    results["GEXshvar116"] = results["GEXshvar115"] / (results["GEXshvar004"] * 3) if results["GEXshvar004"] > 0 else -1
    # 最近6个月连续7日最大流入额/（7*年日均流入额）
    results["GEXshvar119"] = results["GEXshvar118"] / (results["GEXshvar004"] * 7) if results["GEXshvar004"] > 0 else -1
    # 最近6个月连续7日最小流入额/（7*年日均流入额）
    results["GEXshvar122"] = results["GEXshvar121"] / (results["GEXshvar004"] * 7) if results["GEXshvar004"] > 0 else -1

    # 近7天流入为0的天数/近7天的放款额（万元）
    results["GEXshvar061"] = results["GEXshvar061_1"] / results["GEXshvar061_2"] if results["GEXshvar061_2"] > 0 else -1
    # 近7天的放款额（万元）/近7天流入为0的天数
    results["GEXshvar058"] = results["GEXshvar061_2"] / results["GEXshvar061_1"] if results["GEXshvar061_1"] > 0 else -1
    # 近14天流入为0的天数/近14天的放款额（万元）
    results["GEXshvar025"] = results["GEXshvar025_1"] / results["GEXshvar025_2"] if results["GEXshvar025_2"] > 0 else -1
    # 近14天的放款额（万元）/近14天流入为0的天数
    results["GEXshvar024"] = results["GEXshvar025_2"] / results["GEXshvar025_1"] if results["GEXshvar025_1"] > 0 else -1
    # 近1个月流入为0的天数/近一个月的放款额（万元）
    results["GEXshvar029"] = results["GEXshvar029_1"] / results["GEXshvar029_2"] if results["GEXshvar029_2"] > 0 else -1
    # 近一个月的放款额（万元）/近1个月流入为0的天数
    results["GEXshvar064"] = results["GEXshvar029_2"] / results["GEXshvar029_1"] if results["GEXshvar029_1"] > 0 else -1
    # 近3个月流入为0的天数/近3个月的放款额（万元）
    results["GEXshvar043"] = results["GEXshvar043_1"] / results["GEXshvar043_2"] if results["GEXshvar043_2"] > 0 else -1
    # 近3个月放款额（万元）/近3个月流入为0的天数
    results["GEXshvar034"] = results["GEXshvar043_2"] / results["GEXshvar043_1"] if results["GEXshvar043_1"] > 0 else -1

    return results


def deal_store_derive_features(data, today, store_code):
    data = data.copy()

    if isinstance(today, str):
        today = parser.parse(today)

    results = {}

    def deal_lasted_app_count(data, days=7, industry_one=None, industry_two=None):
        """
        申请_全行业_申请件数_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if industry_one:
                curr_app_base = curr_app_base[curr_app_base["C_INDUSTRY_ONE"] == industry_one].reset_index(drop=True)

            if industry_two:
                curr_app_base = curr_app_base[curr_app_base["C_INDUSTRY_TWO"] == industry_two].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base["C_CUST_IDNO"].nunique()
            else:
                return 0
        else:
            return -1

    results["GEXstore036"] = deal_lasted_app_count(data, days=1, industry_two="口腔")
    results["GEXstore037"] = deal_lasted_app_count(data, days=3, industry_two="口腔")
    results["GEXstore038"] = deal_lasted_app_count(data, days=7, industry_two="口腔")

    results["GEXstore039"] = deal_lasted_app_count(data, days=1, industry_two="眼科")
    results["GEXstore040"] = deal_lasted_app_count(data, days=3, industry_two="眼科")
    results["GEXstore041"] = deal_lasted_app_count(data, days=7, industry_two="眼科")

    results["GEXstore042"] = deal_lasted_app_count(data, days=1, industry_two="听力")
    results["GEXstore043"] = deal_lasted_app_count(data, days=3, industry_two="听力")
    results["GEXstore044"] = deal_lasted_app_count(data, days=7, industry_two="听力")

    if data.get("app_base") is not None and len(data["app_base"]) > 0:
        data["app_base"] = data["app_base"].query("N_APP_STATUS in (130, 140, 160)").reset_index()

    def deal_lasted_app_high_age(data, days=7):
        """
        申请_医疗整形_高龄占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_INDUSTRY_ONE"].isin(["医疗", "美容"]))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if curr_app_base["C_CUST_IDNO"].nunique() < 10:
                return -1
            else:
                # curr_app_base = curr_app_base.sort_values("N_AGE", ascending=False).drop_duplicates("C_CUST_IDNO", keep="first")
                curr_app_base["age_status"] = curr_app_base.apply(
                    lambda row: 1 if (row["C_GENDER"] == 0 and row["N_AGE"] >= 40) or (
                            row["C_GENDER"] == 1 and row["N_AGE"] >= 35) else 0, axis=1)
                return curr_app_base[curr_app_base["age_status"] == 1]["C_CUST_IDNO"].nunique() / curr_app_base[
                    "C_CUST_IDNO"].nunique()
        else:
            return -1

    results["GEXstore001"] = deal_lasted_app_high_age(data, days=7)
    results["GEXstore002"] = deal_lasted_app_high_age(data, days=14)
    results["GEXstore003"] = deal_lasted_app_high_age(data, days=30)
    results["GEXstore004"] = deal_lasted_app_high_age(data, days=60)
    results["GEXstore005"] = deal_lasted_app_high_age(data, days=90)

    def deal_lasted_app_low_age(data, days=7):
        """
        申请_医疗整形_低龄占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_INDUSTRY_ONE"].isin(["医疗", "美容"]))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if curr_app_base["C_CUST_IDNO"].nunique() < 10:
                return -1
            else:
                # curr_app_base = curr_app_base.sort_values("N_AGE", ascending=False).drop_duplicates("C_CUST_IDNO", keep="last")
                curr_app_base["age_status"] = curr_app_base.apply(lambda row: 1 if row["N_AGE"] <= 19 else 0, axis=1)
                return curr_app_base[curr_app_base["age_status"] == 1]["C_CUST_IDNO"].nunique() / curr_app_base[
                    "C_CUST_IDNO"].nunique()
        else:
            return -1

    results["GEXstore006"] = deal_lasted_app_low_age(data, days=7)
    results["GEXstore007"] = deal_lasted_app_low_age(data, days=14)
    results["GEXstore008"] = deal_lasted_app_low_age(data, days=30)
    results["GEXstore009"] = deal_lasted_app_low_age(data, days=60)
    results["GEXstore010"] = deal_lasted_app_low_age(data, days=90)

    results["GEXstore011"] = deal_lasted_app_count(data, days=7)
    results["GEXstore012"] = deal_lasted_app_count(data, days=14)
    results["GEXstore013"] = deal_lasted_app_count(data, days=30)
    results["GEXstore014"] = deal_lasted_app_count(data, days=60)
    results["GEXstore015"] = deal_lasted_app_count(data, days=90)

    def deal_lasted_app_amount(data, days=7):
        """
        申请_全行业_申请金额_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base.groupby(["C_CUST_IDNO"])["N_AMT_APPLIED"].max().sum()
            else:
                return 0
        else:
            return -1

    results["GEXstore016"] = deal_lasted_app_amount(data, days=7)
    results["GEXstore017"] = deal_lasted_app_amount(data, days=14)
    results["GEXstore018"] = deal_lasted_app_amount(data, days=30)
    results["GEXstore019"] = deal_lasted_app_amount(data, days=60)
    results["GEXstore020"] = deal_lasted_app_amount(data, days=90)

    def deal_lasted_app_amount_avg(data, days=7, industry_one=None, industry_two=None):
        """
        申请_申请件均_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if industry_one:
                curr_app_base = curr_app_base[curr_app_base["C_INDUSTRY_ONE"] == industry_one].reset_index(drop=True)

            if industry_two:
                curr_app_base = curr_app_base[curr_app_base["C_INDUSTRY_TWO"] == industry_two].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base.groupby(["C_CUST_IDNO"])["N_AMT_APPLIED"].max().mean()
            else:
                return 0
        else:
            return -1

    results["GEXstore021"] = deal_lasted_app_amount_avg(data, days=7)
    results["GEXstore022"] = deal_lasted_app_amount_avg(data, days=14)
    results["GEXstore023"] = deal_lasted_app_amount_avg(data, days=30)
    results["GEXstore024"] = deal_lasted_app_amount_avg(data, days=60)
    results["GEXstore025"] = deal_lasted_app_amount_avg(data, days=90)

    def deal_lasted_app_count_mom(data, days=7):
        """
        申请_全行业_申请件数环比增长_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            app_base = app_base[app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G")].reset_index(drop=True)

            curr_app_base = app_base[app_base["D_APPLICATION"] >= today - relativedelta(days=days)][
                "C_CUST_IDNO"].nunique()
            last_app_base = app_base[(app_base["D_APPLICATION"] >= today - relativedelta(days=2 * days)) & (
                    app_base["D_APPLICATION"] < today - relativedelta(days=days))]["C_CUST_IDNO"].nunique()

            if last_app_base > 0:
                return curr_app_base / last_app_base - 1.00
            else:
                return -99
        else:
            return -99

    results["GEXstore026"] = deal_lasted_app_count_mom(data, days=7)
    results["GEXstore027"] = deal_lasted_app_count_mom(data, days=14)
    results["GEXstore028"] = deal_lasted_app_count_mom(data, days=30)
    results["GEXstore028_1"] = deal_lasted_app_count_mom(data, days=60)
    results["GEXstore029"] = deal_lasted_app_count_mom(data, days=90)

    def deal_lasted_app_amount_mom(data, days=7):
        """
        申请_全行业_申请金额环比增长_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            app_base = app_base[app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G")].reset_index(drop=True)

            curr_app_base = \
                app_base[app_base["D_APPLICATION"] >= today - relativedelta(days=days)].groupby(["C_CUST_IDNO"])[
                    "N_AMT_APPLIED"].max().sum()
            last_app_base = app_base[(app_base["D_APPLICATION"] >= today - relativedelta(days=2 * days)) & (
                    app_base["D_APPLICATION"] < today - relativedelta(days=days))].groupby(["C_CUST_IDNO"])[
                "N_AMT_APPLIED"].max().sum()

            if last_app_base > 0:
                return curr_app_base / last_app_base - 1.00
            else:
                return -99
        else:
            return -99

    results["GEXstore030"] = deal_lasted_app_amount_mom(data, days=7)
    results["GEXstore031"] = deal_lasted_app_amount_mom(data, days=14)
    results["GEXstore032"] = deal_lasted_app_amount_mom(data, days=30)
    results["GEXstore033"] = deal_lasted_app_amount_mom(data, days=60)
    results["GEXstore034"] = deal_lasted_app_amount_mom(data, days=90)

    if data.get("app_base") is not None and len(data["app_base"]) > 0:
        app_base = data["app_base"].copy()
        app_base = app_base[
            app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith(
                "超G")].reset_index(drop=True)

        if len(app_base) > 0:
            results["GEXstore035"] = (today - app_base["D_APPLICATION"].min()).days
        else:
            results["GEXstore035"] = -1
    else:
        results["GEXstore035"] = -1

    def deal_lasted_app_amount_similar(data, days=7):
        """
        申请_全行业_金额雷同订单占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["N_AMT_APPLIED"] % 5000 != 0)
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base.groupby(["C_CUST_IDNO"])["N_AMT_APPLIED"].max().value_counts().iloc[0] / \
                       curr_app_base["C_CUST_IDNO"].nunique()
            else:
                return -1
        else:
            return -1

    results["GEXstore048"] = deal_lasted_app_amount_similar(data, days=7)
    results["GEXstore049"] = deal_lasted_app_amount_similar(data, days=14)
    results["GEXstore050"] = deal_lasted_app_amount_similar(data, days=30)

    def deal_lasted_app_city_anomal(data, days=7):
        """
        申请_全行业_户籍异常订单占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base[
                    ~curr_app_base["C_PERMANENT_CODE"].isnull()
                    & ~curr_app_base["C_STORE_CITY_CODE"].isnull()
                    & (curr_app_base["C_PERMANENT_CODE"].str[:4] != curr_app_base["C_STORE_CITY_CODE"].str[:4])
                    ]["C_CUST_IDNO"].nunique()
            else:
                return -1
        else:
            return -1

    results["GEXstore051"] = deal_lasted_app_city_anomal(data, days=7)
    results["GEXstore052"] = deal_lasted_app_city_anomal(data, days=14)
    results["GEXstore053"] = deal_lasted_app_city_anomal(data, days=30)

    def deal_lasted_male_plastic(data, days=7):
        """
        申请_整形_男性比例_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if curr_app_base["C_CUST_IDNO"].nunique() < 10:
                return -1
            else:
                return curr_app_base[curr_app_base["C_INDUSTRY_TWO"] == "整形"]["C_CUST_IDNO"].nunique() / curr_app_base[
                    "C_CUST_IDNO"].nunique()
        else:
            return -1

    results["GEXstore054"] = deal_lasted_app_city_anomal(data, days=7)
    results["GEXstore055"] = deal_lasted_app_city_anomal(data, days=14)
    results["GEXstore056"] = deal_lasted_app_city_anomal(data, days=30)

    def deal_lasted_plastic_injection(data, days=7):
        """
        申请_整形_异常针剂比例占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_INDUSTRY_TWO"] == "整形")
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if len(curr_app_base) > 0:
                return curr_app_base[
                           curr_app_base["C_ITEM_NAME"].fillna("").apply(lambda x: True if re.match(
                               r"玻尿酸|除皱瘦脸|皮肤美容|面部美容|抗衰|抗初老|激光脱毛|热玛吉|美白|嫩肤|净肤|美肤|注射|护理|补水|童颜针", x) else False)
                           & (curr_app_base["N_AMT_APPLIED"] >= 15000)
                           ]["C_CUST_IDNO"].nunique() / curr_app_base["C_CUST_IDNO"].nunique()
            else:
                return -1
        else:
            return -1

    results["GEXstore045"] = deal_lasted_plastic_injection(data, days=7)
    results["GEXstore046"] = deal_lasted_plastic_injection(data, days=14)
    results["GEXstore047"] = deal_lasted_plastic_injection(data, days=30)

    def deal_lasted_geodesic_anomal(data, days=7):
        """
        申请_全行业_GPS偏移占比_近N天
        """
        if data["store_info"][["C_LONGITUDE", "C_LATITUDE"]].iloc[0, :].astype(float).isnull().sum() > 0:
            return -1
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            curr_app_base = app_base[
                (app_base["D_APPLICATION"] >= today - relativedelta(days=days))
                & (app_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | app_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if curr_app_base["C_CUST_IDNO"].nunique() < 10:
                return -1
            else:
                store_degree = data["store_info"][["C_LONGITUDE", "C_LATITUDE"]].iloc[0, :].astype(float).to_dict()
                return curr_app_base[
                           curr_app_base[["C_LONGITUDE", "C_LATITUDE"]].astype(float).replace(0.0, np.nan).apply(
                               lambda row: geodesic(
                                   (row["C_LONGITUDE"], row["C_LATITUDE"]),
                                   (store_degree["C_LATITUDE"], store_degree["C_LONGITUDE"])
                               ).km if pd.isnull(row).sum() == 0 else np.nan
                               , axis=1) > 3
                           ]["C_CUST_IDNO"].nunique() / curr_app_base["C_CUST_IDNO"].nunique()
        else:
            return -1

    results["GEXstore057"] = deal_lasted_geodesic_anomal(data, days=7)
    results["GEXstore058"] = deal_lasted_geodesic_anomal(data, days=14)
    results["GEXstore059"] = deal_lasted_geodesic_anomal(data, days=30)

    def deal_lasted_app_score_avg(data, days=7):
        """
        申请_全行业_申请平均模型分_近N天
        """
        score_name_map = {"口腔": "X517", "祛痘": "X475", "植发": "X509", "整形": "X506", "整形": "X612", "皮肤科": "X522",
                          "祛痘": "X631", "职培": "X492"}
        if data.get("score_data") is not None and len(data["score_data"]) > 0:
            score_data = data["score_data"].copy()
            for col in score_name_map.values():
                score_data[col] = score_data[col].astype(float, errors='ignore')
            store_industry = data["store_info"]["C_INDUSTRY_TWO"].tolist()[0]

            curr_score_data = score_data[score_data["D_APPLICATION"] >= today - relativedelta(days=days)].reset_index(
                drop=True)
            score_name = score_name_map.get(store_industry)

            if score_name is not None and curr_score_data[score_name].sum() > 0:
                return curr_score_data[score_name].mean()
            else:
                return -1
        else:
            return -1

    results["GEXstore065"] = deal_lasted_app_score_avg(data, days=7)
    results["GEXstore066"] = deal_lasted_app_score_avg(data, days=14)
    results["GEXstore067"] = deal_lasted_app_score_avg(data, days=30)
    results["GEXstore068"] = deal_lasted_app_score_avg(data, days=60)
    results["GEXstore069"] = deal_lasted_app_score_avg(data, days=90)

    def deal_lasted_loan_count(data, days=7):
        """
        放款_全行业_放款件数_近N天
        """
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            curr_loan_base = loan_base[
                (loan_base["D_LOAN_DATE"] >= today - relativedelta(days=days))
                & (loan_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | loan_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            if len(curr_loan_base) > 0:
                return curr_loan_base["C_EXT_ID"].nunique()
            else:
                return 0
        else:
            return -1

    results["GEXstore070"] = deal_lasted_loan_count(data, days=7)
    results["GEXstore071"] = deal_lasted_loan_count(data, days=14)
    results["GEXstore072"] = deal_lasted_loan_count(data, days=30)
    results["GEXstore073"] = deal_lasted_loan_count(data, days=60)
    results["GEXstore074"] = deal_lasted_loan_count(data, days=90)

    def deal_lasted_loan_amount(data, days=7):
        """
        放款_全行业_放款金额_近N天
        """
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            curr_loan_base = loan_base[
                (loan_base["D_LOAN_DATE"] >= today - relativedelta(days=days))
                & (loan_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | loan_base[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                ].reset_index(drop=True)

            return curr_loan_base["N_LOAN_AMOUNT"].sum()
        else:
            return -1

    results["GEXstore075"] = deal_lasted_loan_amount(data, days=7)
    results["GEXstore076"] = deal_lasted_loan_amount(data, days=14)
    results["GEXstore077"] = deal_lasted_loan_amount(data, days=30)
    results["GEXstore078"] = deal_lasted_loan_amount(data, days=60)
    results["GEXstore079"] = deal_lasted_loan_amount(data, days=90)

    def deal_lasted_loan_count_mom(data, days=7):
        """
        放款_全行业_放款件数环比增长_近N天
        """
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            loan_base = loan_base[loan_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | loan_base[
                "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G")].reset_index(drop=True)

            curr_loan_base = len(loan_base[loan_base["D_LOAN_DATE"] >= today - relativedelta(days=days)])
            last_loan_base = len(loan_base[(loan_base["D_LOAN_DATE"] >= today - relativedelta(days=2 * days)) & (
                    loan_base["D_LOAN_DATE"] < today - relativedelta(days=days))])

            if last_loan_base > 0:
                return curr_loan_base / last_loan_base - 1.00
            else:
                return -99
        else:
            return -99

    results["GEXstore080"] = deal_lasted_loan_count_mom(data, days=7)
    results["GEXstore081"] = deal_lasted_loan_count_mom(data, days=14)
    results["GEXstore082"] = deal_lasted_loan_count_mom(data, days=30)
    results["GEXstore083"] = deal_lasted_loan_count_mom(data, days=60)
    results["GEXstore084"] = deal_lasted_loan_count_mom(data, days=90)

    def deal_lasted_loan_amount_mom(data, days=7):
        """
        放款_全行业_放款件数环比增长_近N天
        """
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            loan_base = loan_base[loan_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | loan_base[
                "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G")].reset_index(drop=True)

            curr_loan_base = loan_base[loan_base["D_LOAN_DATE"] >= today - relativedelta(days=days)][
                "N_LOAN_AMOUNT"].sum()
            last_loan_base = loan_base[(loan_base["D_LOAN_DATE"] >= today - relativedelta(days=2 * days)) & (
                    loan_base["D_LOAN_DATE"] < today - relativedelta(days=days))]["N_LOAN_AMOUNT"].sum()

            if last_loan_base > 0:
                return curr_loan_base / last_loan_base - 1.00
            else:
                return -99
        else:
            return -99

    results["GEXstore085"] = deal_lasted_loan_amount_mom(data, days=7)
    results["GEXstore086"] = deal_lasted_loan_amount_mom(data, days=14)
    results["GEXstore087"] = deal_lasted_loan_amount_mom(data, days=30)
    results["GEXstore088"] = deal_lasted_loan_amount_mom(data, days=60)
    results["GEXstore089"] = deal_lasted_loan_amount_mom(data, days=90)

    def deal_current_loan_count(data):
        if data.get("loan_base") is not None and len(data["loan_base"]) > 0:
            loan_base = data["loan_base"].copy()
            loan_base = loan_base[loan_base["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | loan_base[
                "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G")].reset_index(drop=True)

            return loan_base["D_SETTLE_DATE"].isnull().sum()
        else:
            return 0

    results["GEXstore105"] = deal_current_loan_count(data)

    def deal_replay_pay_amount(data, days=7):
        if data.get("repay_plan") is not None and len(data["repay_plan"]) > 0:
            repay_plan = data["repay_plan"].copy()
            repay_plan = repay_plan[
                (repay_plan["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | repay_plan[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                & (repay_plan["D_FINISH_DATE"].fillna(today).apply(lambda x: (today - x).days) <= days)
                & (repay_plan["D_PAY_DATE"] >= today - relativedelta(days=days))
                & (repay_plan["D_PAY_DATE"] <= today)
                ].reset_index(drop=True)

            return repay_plan["N_ALL_PAY_CORPUS"].sum()
        else:
            return 0

    results["GEXstore116"] = deal_replay_pay_amount(data, days=7)
    results["GEXstore117"] = deal_replay_pay_amount(data, days=14)
    results["GEXstore118"] = deal_replay_pay_amount(data, days=30)
    results["GEXstore119"] = deal_replay_pay_amount(data, days=60)
    results["GEXstore120"] = deal_replay_pay_amount(data, days=90)

    def deal_replay_overdue_ratio(data, days=7, overdue=1, obs_days=0):
        if data.get("repay_plan") is not None and len(data["repay_plan"]) > 0:
            repay_plan = data["repay_plan"].copy()
            curr_repay_plan = repay_plan[
                (repay_plan["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | repay_plan[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                & (repay_plan["D_FINISH_DATE"].fillna(today).apply(lambda x: (today - x).days) <= days)
                & (repay_plan["D_PAY_DATE"] >= today - relativedelta(days=days))
                & (repay_plan["D_PAY_DATE"] <= today - relativedelta(days=obs_days))
                ].reset_index(drop=True)

            clear_amount = curr_repay_plan["N_ALL_PAY_CORPUS"].sum()
            overdue_clear_amount = curr_repay_plan[
                (curr_repay_plan["D_FINISH_DATE"].fillna(today) - curr_repay_plan["D_PAY_DATE"]).dt.days >= overdue][
                "N_ALL_PAY_CORPUS"].sum()

            if clear_amount > 0:
                return overdue_clear_amount / clear_amount
            else:
                return -1
        else:
            return -1

    results["GEXstore121"] = deal_replay_overdue_ratio(data, days=7, overdue=1)
    results["GEXstore122"] = deal_replay_overdue_ratio(data, days=14, overdue=1)
    results["GEXstore123"] = deal_replay_overdue_ratio(data, days=30, overdue=1)
    results["GEXstore124"] = deal_replay_overdue_ratio(data, days=60, overdue=1)
    results["GEXstore125"] = deal_replay_overdue_ratio(data, days=90, overdue=1)
    results["GEXstore126"] = deal_replay_overdue_ratio(data, days=7, overdue=4, obs_days=4)
    results["GEXstore127"] = deal_replay_overdue_ratio(data, days=14, overdue=4, obs_days=4)
    results["GEXstore128"] = deal_replay_overdue_ratio(data, days=30, overdue=4, obs_days=4)
    results["GEXstore129"] = deal_replay_overdue_ratio(data, days=60, overdue=4, obs_days=4)
    results["GEXstore130"] = deal_replay_overdue_ratio(data, days=90, overdue=4, obs_days=4)

    def deal_replay_overdue_collection_ratio(data, days=7, overdue=1, obs_days=0):
        if data.get("repay_plan") is not None and len(data["repay_plan"]) > 0:
            repay_plan = data["repay_plan"].copy()
            curr_repay_plan = repay_plan[
                (repay_plan["C_FINANCE_PRODUCT_TYPE"].str.startswith("无卡") | repay_plan[
                    "C_FINANCE_PRODUCT_TYPE"].str.startswith("超G"))
                & (repay_plan["D_FINISH_DATE"].fillna(today).apply(lambda x: (today - x).days) <= days)
                & (repay_plan["D_PAY_DATE"] >= today - relativedelta(days=days))
                & (repay_plan["D_PAY_DATE"] <= today - relativedelta(days=obs_days))
                & (repay_plan["D_FINISH_DATE"].fillna(today) - repay_plan["D_PAY_DATE"]).dt.days >= 1
                ].reset_index(drop=True)

            clear_amount = curr_repay_plan["N_ALL_PAY_CORPUS"].sum()
            overdue_clear_amount = curr_repay_plan[
                (curr_repay_plan["D_FINISH_DATE"].fillna(today) - curr_repay_plan["D_PAY_DATE"]).dt.days >= overdue][
                "N_ALL_PAY_CORPUS"].sum()

            if clear_amount > 0:
                return 1 - overdue_clear_amount / clear_amount
            else:
                return -1
        else:
            return -1

    results["GEXstore131"] = deal_replay_overdue_collection_ratio(data, days=30, overdue=4, obs_days=4)
    results["GEXstore132"] = deal_replay_overdue_collection_ratio(data, days=60, overdue=4, obs_days=4)
    results["GEXstore133"] = deal_replay_overdue_collection_ratio(data, days=90, overdue=4, obs_days=4)
    results["GEXstore134"] = deal_replay_overdue_collection_ratio(data, days=180, overdue=4, obs_days=4)

    results["GEXstore135"] = deal_replay_overdue_collection_ratio(data, days=30, overdue=10, obs_days=10)
    results["GEXstore136"] = deal_replay_overdue_collection_ratio(data, days=60, overdue=10, obs_days=10)
    results["GEXstore137"] = deal_replay_overdue_collection_ratio(data, days=90, overdue=10, obs_days=10)
    results["GEXstore138"] = deal_replay_overdue_collection_ratio(data, days=180, overdue=10, obs_days=10)

    def deal_bill_upload_fail_ratio(data, days=7):
        """
        申请_全行业_上传小票失败占比_近N天
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            bill_upload_data = data["bill_upload_data"].copy()
            app_base = data["app_base"].copy()

            curr_app_base = app_base[app_base["D_APPLICATION"] >= today - relativedelta(days=days)].reset_index(
                drop=True)
            curr_bill_upload_data = bill_upload_data[
                bill_upload_data["D_APPLY_TIME"] >= today - relativedelta(days=days)]

            if curr_app_base["C_CUST_IDNO"].nunique() < 10:
                return -1
            else:
                return curr_bill_upload_data["C_ID_NO"].nunique() / curr_app_base["C_CUST_IDNO"].nunique()
        else:
            return -1

    results["GEXstore060"] = deal_bill_upload_fail_ratio(data, days=7)
    results["GEXstore061"] = deal_bill_upload_fail_ratio(data, days=14)
    results["GEXstore062"] = deal_bill_upload_fail_ratio(data, days=30)
    results["GEXstore063"] = deal_bill_upload_fail_ratio(data, days=60)
    results["GEXstore064"] = deal_bill_upload_fail_ratio(data, days=90)

    def deal_lasted_store_complaint(data, days=7):
        """
        运营_全行业_投诉数量_近N天
        """
        if data.get("store_complaint") is not None and len(data["store_complaint"]) > 0:
            store_complaint = data["store_complaint"].copy()

            curr_loan_base = store_complaint[store_complaint["D_CREATE"] >= today - relativedelta(days=days)]

            if len(curr_loan_base) > 0:
                return curr_loan_base["C_APP_ID"].nunique()
            else:
                return 0
        else:
            return -1

    results["GEXstore106"] = deal_lasted_store_complaint(data, days=7)
    results["GEXstore107"] = deal_lasted_store_complaint(data, days=14)
    results["GEXstore108"] = deal_lasted_store_complaint(data, days=30)
    results["GEXstore109"] = deal_lasted_store_complaint(data, days=60)
    results["GEXstore110"] = deal_lasted_store_complaint(data, days=90)

    def deal_lasted_store_complaint_ratio(data, days=7):
        """
        运营_全行业_投诉占比_近N天
        """
        if data.get("store_complaint") is not None and len(data["store_complaint"]) > 0:
            store_complaint = data["store_complaint"].copy()

            curr_loan_base = store_complaint[store_complaint["D_CREATE"] >= today - relativedelta(days=days)]

            # 门店在贷订单数
            curr_store_loan_count = data["loan_base"]["D_SETTLE_DATE"].isnull().sum()

            if len(curr_loan_base) > 0 and curr_store_loan_count > 0:
                return curr_loan_base["C_APP_ID"].nunique() / curr_store_loan_count
            else:
                return -1
        else:
            return -1

    results["GEXstore111"] = deal_lasted_store_complaint_ratio(data, days=7)
    results["GEXstore112"] = deal_lasted_store_complaint_ratio(data, days=14)
    results["GEXstore113"] = deal_lasted_store_complaint_ratio(data, days=30)
    results["GEXstore114"] = deal_lasted_store_complaint_ratio(data, days=60)
    results["GEXstore115"] = deal_lasted_store_complaint_ratio(data, days=90)

    def deal_lasted_store_freeze_count(data, days=7):
        """
        运营_门店冻结次数_近N天
        """
        if data.get("store_freeze") is not None and len(data["store_freeze"]) > 0:
            store_freeze = data["store_freeze"].copy()

            curr_store_freeze = store_freeze[store_freeze["D_BATCH_DATE"] >= today - relativedelta(days=days)]
            return len(curr_store_freeze)
        else:
            return -1

    results["GEXstore156"] = deal_lasted_store_freeze_count(data, days=7)
    results["GEXstore157"] = deal_lasted_store_freeze_count(data, days=14)
    results["GEXstore158"] = deal_lasted_store_freeze_count(data, days=30)
    results["GEXstore159"] = deal_lasted_store_freeze_count(data, days=60)
    results["GEXstore160"] = deal_lasted_store_freeze_count(data, days=90)
    results["GEXstore161"] = deal_lasted_store_freeze_count(data, days=180)
    results["GEXstore162"] = deal_lasted_store_freeze_count(data, days=360)

    def deal_lasted_order_freeze_count(data, days=7):
        """
        运营_订单冻结次数_近N天
        """
        if data.get("order_freeze") is not None and len(data["order_freeze"]) > 0:
            order_freeze = data["order_freeze"].copy()

            curr_order_freeze = order_freeze[order_freeze["D_FREEZE_TIME"] >= today - relativedelta(days=days)]
            return curr_order_freeze["C_APP_ID"].nunique()
        else:
            return -1

    results["GEXstore163"] = deal_lasted_order_freeze_count(data, days=7)
    results["GEXstore164"] = deal_lasted_order_freeze_count(data, days=14)
    results["GEXstore165"] = deal_lasted_order_freeze_count(data, days=30)
    results["GEXstore166"] = deal_lasted_order_freeze_count(data, days=60)
    results["GEXstore167"] = deal_lasted_order_freeze_count(data, days=90)
    results["GEXstore168"] = deal_lasted_order_freeze_count(data, days=180)
    results["GEXstore169"] = deal_lasted_order_freeze_count(data, days=360)

    def deal_lasted_order_losing_amount(data, days=7):
        """
        贷中_全行业_失联订单金额_近N天
        """
        if data.get("collection_data") is not None and len(data["collection_data"]) > 0:
            collection_data = data["collection_data"].copy()

            curr_collection_data = collection_data[
                collection_data["D_INCOLLECTION_DATE"] >= today - relativedelta(days=days)]
            return curr_collection_data["N_CURR_PAY_AMOUNT"].sum()
        else:
            return -1

    results["GEXstore139"] = deal_lasted_order_losing_amount(data, days=7)
    results["GEXstore140"] = deal_lasted_order_losing_amount(data, days=14)
    results["GEXstore141"] = deal_lasted_order_losing_amount(data, days=30)
    results["GEXstore142"] = deal_lasted_order_losing_amount(data, days=60)
    results["GEXstore143"] = deal_lasted_order_losing_amount(data, days=90)

    def deal_lasted_order_due_amount(data, days=7):
        """
        近N天天出账账单的出账本金总和
        """
        if data.get("repay_plan") is not None and len(data["repay_plan"]) > 0:
            repay_plan = data["repay_plan"].copy()

            curr_repay_plan = repay_plan[repay_plan["D_PAY_DATE"] >= today - relativedelta(days=days)]
            return curr_repay_plan["N_ALL_PAY_CORPUS"].sum()
        else:
            return -1

    results["GEXstore144_1"] = deal_lasted_order_due_amount(data, days=7)
    results["GEXstore145_1"] = deal_lasted_order_due_amount(data, days=14)
    results["GEXstore146_1"] = deal_lasted_order_due_amount(data, days=30)
    results["GEXstore147_1"] = deal_lasted_order_due_amount(data, days=60)
    results["GEXstore148_1"] = deal_lasted_order_due_amount(data, days=90)

    def deal_lasted_order_fraud_count(data, days=7):
        """
        运营_欺诈次数_近N天
        """
        if data.get("order_report") is not None and len(data["order_report"]) > 0:
            order_report = data["order_report"].copy()

            curr_order_report = order_report[order_report["D_END_TIME"] >= today - relativedelta(days=days)]
            return curr_order_report["C_APP_ID"].nunique()
        else:
            return -1

    results["GEXstore149"] = deal_lasted_order_fraud_count(data, days=7)
    results["GEXstore150"] = deal_lasted_order_fraud_count(data, days=14)
    results["GEXstore151"] = deal_lasted_order_fraud_count(data, days=30)
    results["GEXstore152"] = deal_lasted_order_fraud_count(data, days=60)
    results["GEXstore153"] = deal_lasted_order_fraud_count(data, days=90)
    results["GEXstore154"] = deal_lasted_order_fraud_count(data, days=180)
    results["GEXstore155"] = deal_lasted_order_fraud_count(data, days=360)

    def deal_lasted_apply_rate(data, days=7):
        """
        近N个月-客户通过率
        """
        if data.get("app_base") is not None and len(data["app_base"]) > 0:
            app_base = data["app_base"].copy()
            total_apply = app_base["N_APP_STATUS"].isin([130, 140, 160]).sum()

            if total_apply > 0:
                return app_base["N_APP_STATUS"].isin([130, 160]).sum() / total_apply
            else:
                return 0
        else:
            return -1

    results["GEXstore181"] = deal_lasted_apply_rate(data, days=30)
    results["GEXstore175"] = deal_lasted_apply_rate(data, days=90)
    results["GEXstore178"] = deal_lasted_apply_rate(data, days=180)

    def deal_var_ym(data):
        result = {}
        if data.get("repay_plan") is not None and len(data["repay_plan"]) > 0:
            repay_plan = data["repay_plan"].copy()
            loan_base = data["loan_base"].copy()
            repay_plan['D_SETTLE_DATE'] = repay_plan['C_EXT_ID'].apply(
                lambda x: loan_base[loan_base['C_EXT_ID'] == x]['D_SETTLE_DATE'].max())
            repay_plan_yet = repay_plan[repay_plan['D_SETTLE_DATE'].isnull()]

            # 存量在贷-前3个月至前9个月（跨度6个月）放款-坏客户比例（fpd-vintage30+）
            temp = repay_plan_yet[(repay_plan_yet['D_LOAN_DATE'] < today - relativedelta(months=3) + relativedelta(day=1)) & (repay_plan_yet['D_LOAN_DATE'] >= today - relativedelta(months=9) + relativedelta(day=1))]
            fpd_dict = {C_EXT_ID: (
                sub_df.sort_values(['N_CURR_TENOR'], ascending=True).iloc[0, :]['N_OVERDUE_DAYS'].max() if sub_df.shape[
                                                                                                               0] >= 1 else 0)
                for C_EXT_ID, sub_df in temp.groupby('C_EXT_ID')}
            result['GEXstore170'] = sum([int(x >= 30) for x in fpd_dict.values()]) / (temp.shape[0] + 1e-6)

            # 无卡-存量在贷-前3个月至前15个月（跨度12个月）放款-坏客户比例（fpd-vintage30+）
            temp = repay_plan_yet[
                (repay_plan_yet['D_LOAN_DATE'] < today - relativedelta(months=3) + relativedelta(day=1)) &
                (repay_plan_yet['D_LOAN_DATE'] >= today - relativedelta(months=15) + relativedelta(day=1)) &
                (repay_plan_yet['C_FINANCE_PRODUCT_TYPE'].str.contains('无卡'))
                ]
            fpd_dict = {C_EXT_ID: (
                sub_df.sort_values(['N_CURR_TENOR'], ascending=True).iloc[0, :]['N_OVERDUE_DAYS'].max() if sub_df.shape[
                                                                                                               0] >= 1 else 0)
                for C_EXT_ID, sub_df in temp.groupby('C_EXT_ID')}
            result['GEXstore182'] = sum([int(x >= 30) for x in fpd_dict.values()]) / (temp.shape[0] + 1e-6)

            # 自付-存量在贷-前3个月至前15个月（跨度12个月）放款-坏客户比例（fpd-vintage30+）
            temp = repay_plan_yet[
                (repay_plan_yet['D_LOAN_DATE'] < today - relativedelta(months=3) + relativedelta(day=1)) &
                (repay_plan_yet['D_LOAN_DATE'] >= today - relativedelta(months=15) + relativedelta(day=1)) &
                (repay_plan_yet['C_FINANCE_PRODUCT_TYPE'].str.contains('自付'))
                ]
            fpd_dict = {C_EXT_ID: (
                sub_df.sort_values(['N_CURR_TENOR'], ascending=True).iloc[0, :]['N_OVERDUE_DAYS'].max() if sub_df.shape[
                                                                                                               0] >= 1 else 0)
                for C_EXT_ID, sub_df in temp.groupby('C_EXT_ID')}
            result['GEXstore183'] = sum([int(x >= 30) for x in fpd_dict.values()]) / (temp.shape[0] + 1e-6)

            # 存量在贷-前5个月至前11个月放款-坏客户比例（mob3-vintage30+）
            temp = repay_plan_yet[
                (repay_plan_yet['D_LOAN_DATE'] < today - relativedelta(months=5) + relativedelta(day=1)) &
                (repay_plan_yet['D_LOAN_DATE'] >= today - relativedelta(months=11) + relativedelta(
                    day=1))]
            fpd_dict = {C_EXT_ID: (
                sub_df.sort_values(['N_CURR_TENOR'], ascending=True).iloc[2, :]['N_OVERDUE_DAYS'].max() if sub_df.shape[
                                                                                                               0] >= 3 else 0)
                for C_EXT_ID, sub_df in temp.groupby('C_EXT_ID')}
            result['GEXstore171'] = sum([int(x >= 30) for x in fpd_dict.values()]) / (temp.shape[0] + 1e-6)

            # 存量在贷-前8个月至前14个月放款-坏客户比例（mob3-vintage30+）
            temp = repay_plan_yet[
                (repay_plan_yet['D_LOAN_DATE'] < today - relativedelta(months=8) + relativedelta(day=1)) &
                (repay_plan_yet['D_LOAN_DATE'] >= today - relativedelta(months=14) + relativedelta(
                    day=1))]
            fpd_dict = {C_EXT_ID: (
                sub_df.sort_values(['N_CURR_TENOR'], ascending=True).iloc[2, :]['N_OVERDUE_DAYS'].max() if sub_df.shape[
                                                                                                               0] >= 3 else 0)
                for C_EXT_ID, sub_df in temp.groupby('C_EXT_ID')}
            result['GEXstore172'] = sum([int(x >= 30) for x in fpd_dict.values()]) / (temp.shape[0] + 1e-6)
        else:
            result['GEXstore170'] = -1
            result['GEXstore182'] = -1
            result['GEXstore183'] = -1
            result['GEXstore171'] = -1
            result['GEXstore172'] = -1

        app_base = data["app_base"].copy()
        if app_base.shape[0] > 0:

            celue_reject = app_base[app_base['C_REJECT_TYPE'] == '策略拒绝']
            if_score = celue_reject['C_REJECT_REASON'].apply(
                lambda x: not pd.isnull(x) and re.search('分|XGB', re.sub('分流|分期|分类', '', x)) is not None and re.search('人脸', x) is None)

            celue_reject_temp = celue_reject[
                (celue_reject['D_APPLICATION'] >= today - relativedelta(days=90)) & (
                        celue_reject['D_APPLICATION'] < today)]
            app_base_temp = app_base[
                (app_base['D_APPLICATION'] >= today - relativedelta(days=90)) & (app_base['D_APPLICATION'] < today)]
            # 近3个月-客户策略拒绝率-评分
            result['GEXstore173'] = celue_reject_temp[if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1
            # 近3个月-客户策略拒绝率-硬规则
            result['GEXstore174'] = celue_reject_temp[~if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1

            celue_reject_temp = celue_reject[(celue_reject['D_APPLICATION'] >= today - relativedelta(days=180)) & (
                    celue_reject['D_APPLICATION'] < today)]
            app_base_temp = app_base[
                (app_base['D_APPLICATION'] >= today - relativedelta(days=180)) & (app_base['D_APPLICATION'] < today)]
            # 近6个月-客户策略拒绝率-评分
            result['GEXstore176'] = celue_reject_temp[if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1
            # 近6个月-客户策略拒绝率-硬规则
            result['GEXstore177'] = celue_reject_temp[~if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1

            celue_reject_temp = celue_reject[
                (celue_reject['D_APPLICATION'] >= today - relativedelta(days=30)) & (
                        celue_reject['D_APPLICATION'] < today)]
            app_base_temp = app_base[
                (app_base['D_APPLICATION'] >= today - relativedelta(days=30)) & (app_base['D_APPLICATION'] < today)]
            # 近6个月-客户策略拒绝率-评分
            result['GEXstore179'] = celue_reject_temp[if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1
            # 近6个月-客户策略拒绝率-硬规则
            result['GEXstore180'] = celue_reject_temp[~if_score].shape[0] / app_base_temp.shape[0] if \
                app_base_temp.shape[
                    0] != 0 else -1
        else:
            result['GEXstore173'] = -1
            result['GEXstore174'] = -1
            result['GEXstore176'] = -1
            result['GEXstore177'] = -1
            result['GEXstore179'] = -1
            result['GEXstore180'] = -1

        return result

    result_ym = deal_var_ym(data)
    results.update(result_ym)

    results["GEXstore144"] = results["GEXstore139"] / results["GEXstore144_1"] if results["GEXstore144_1"] > 0 and \
                                                                                  results["GEXstore139"] >= 0 else -1
    results["GEXstore145"] = results["GEXstore140"] / results["GEXstore145_1"] if results["GEXstore145_1"] > 0 and \
                                                                                  results["GEXstore140"] >= 0 else -1
    results["GEXstore146"] = results["GEXstore141"] / results["GEXstore146_1"] if results["GEXstore146_1"] > 0 and \
                                                                                  results["GEXstore141"] >= 0 else -1
    results["GEXstore147"] = results["GEXstore142"] / results["GEXstore147_1"] if results["GEXstore147_1"] > 0 and \
                                                                                  results["GEXstore142"] >= 0 else -1
    results["GEXstore148"] = results["GEXstore143"] / results["GEXstore148_1"] if results["GEXstore148_1"] > 0 and \
                                                                                  results["GEXstore143"] >= 0 else -1
    return results


global results
results = []


@atexit.register
def cache_results():
    global results
    if len(results) > 0:
        save_pickle(results, "./parsed_results.pkl")


if __name__ == '__main__':
    # 配置相关参数或环境
    # 命令行传参
    parser = argparse.ArgumentParser(description="商户变量解析相关参数配置")
    parser.add_argument('--max_workers', type=int, default=4, help="执行解析的线程池个数")
    parser.add_argument('--seed', type=int, default=3407, help="随机种子, 保证结果可复现时设置")
    parser.add_argument('--database', type=str, default="ZHANGLUPING", help="最终结果保存的数据库")
    parser.add_argument('--table_name', type=str, default="DM_MID_STORE_FEATURES", help="最终结果保存的表名称")
    parser.add_argument('--inf', type=float, default=999999999, help="无穷大对应的数值")
    parser.add_argument('--batch_size', type=int, default=256, help="每次查询数据库的商户数量")
    parser.add_argument('--replace', action='store_true', default=False, help="是否替换已有的数据库表")
    parser.add_argument('--clear', action='store_true', default=False, help="是否清除本地缓存的解析数据")
    parser.add_argument('--unlock', action='store_true', default=False, help="是否清除文件锁")
    parser.add_argument('--debug', action='store_true', default=False, help="是否开启DEBUG模式，只执行几家门店的变量解析")
    parser.add_argument('--parse', type=str, default="all", help="需要解析的变量，默认 all，解析商户变量和收单变量，可选 zjj、derive、all")
    parser.add_argument('--traceback', action='store_true', default=False, help="是否为数据回溯，回溯时读取excel文件中的店铺编号和日期进行回溯")
    args = parser.parse_args()

    args.zjj_status = True if args.parse in ("all", "zjj") else False
    args.derive_status = True if args.parse in ("all", "derive") else False

    if args.traceback:
        args.table_name += "_TB"
    elif args.debug:
        args.table_name += "_DB"

    seed_everything(args.seed)

    # 全局文件锁, 仅允许存在一个解析的程序
    if args.unlock is False and os.path.isfile("parse_lock"):
        sys.exit()
    else:
        with open("parse_lock", 'w') as f:
            f.write("lock")

    all_store_data = query_all_store_data.query()
    all_store_data = all_store_data.dropna(how="all")

    # 获取所有商户的基础信息表
    if args.traceback:
        traceback_store_data = query_traceback_store_data("feature_calculator/门店坏账及盈利含回溯_202304.xlsx")
        all_store_traceback_date = dict(zip(traceback_store_data["C_STORE_CODE"], traceback_store_data["D_CONCLUSION_TIME"]))

    def get_store_data(batch_data, store_code, today):
        result = {}
        temp = batch_data["store_info"].copy()
        result["store_info"] = temp[temp["C_STORE_CODE"] == store_code].reset_index(drop=True)
        temp = batch_data["app_base"].copy()
        result["app_base"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_APPLICATION"] <= today)].reset_index(drop=True)
        temp = batch_data["score_data"].copy()
        result["score_data"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_APPLICATION"] <= today)].reset_index(drop=True)
        temp = batch_data["loan_base"].copy()
        result["loan_base"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_LOAN_DATE"] <= today)].reset_index(drop=True)
        temp = batch_data["bill_upload_data"].copy()
        result["bill_upload_data"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_APPLY_TIME"] <= today)].reset_index(drop=True)
        temp = batch_data["store_freeze"].copy()
        result["store_freeze"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_BATCH_DATE"] <= today)].reset_index(drop=True)
        temp = batch_data["order_freeze"].copy()
        result["order_freeze"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_FREEZE_TIME"] <= today)].reset_index(drop=True)
        temp = batch_data["store_complaint"].copy()
        result["store_complaint"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_CREATE"] <= today)].reset_index(drop=True)
        temp = batch_data["collection_data"].copy()
        result["collection_data"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_INCOLLECTION_DATE"] <= today)].reset_index(drop=True)
        temp = batch_data["repay_plan"].copy()
        result["repay_plan"] = temp[temp["C_EXT_ID"].isin(result["loan_base"]["C_EXT_ID"])].reset_index(drop=True)
        # 计算结清日期
        if result["repay_plan"] is not None and len(result["repay_plan"]) > 0:
            result["repay_plan"]["D_FINISH_DATE"] = result["repay_plan"]["D_FINISH_DATE"].apply(lambda d: np.nan if pd.isnull(d) or d.strftime("%Y-%m-%d") > today else d)
        temp = batch_data["cci_order"].copy()
        result["cci_order"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_CREATE_TIME"] <= today)].reset_index(drop=True)
        temp = batch_data["cci_refund"].copy()
        result["cci_refund"] = temp[(temp["C_STORE_CODE"] == store_code) & (temp["D_UPDATE_TIME"] <= today)].reset_index(drop=True)

        return result


    def deal_store_features(store_codes, all_store_data=None, today=None):
        results = []

        # 查询门店相关数据信息
        batch_data = dict(
            store_info=query_store_info.query(store_code=str(store_codes)[1:-1]),
            app_base=query_app_base.query(store_code=str(store_codes)[1:-1]),
            score_data=query_score_data.query(store_code=str(store_codes)[1:-1]),
            loan_base=query_loan_base.query(store_code=str(store_codes)[1:-1]),
            bill_upload_data=query_bill_upload_data.query(store_code=str(store_codes)[1:-1]),
            store_freeze=query_store_freeze.query(store_code=str(store_codes)[1:-1]),
            order_freeze=query_order_freeze.query(store_code=str(store_codes)[1:-1]),
            store_complaint=query_store_complaint.query(store_code=str(store_codes)[1:-1]),
            collection_data=query_collection_data.query(store_code=str(store_codes)[1:-1]),
            order_report=query_order_report.query(store_code=str(store_codes)[1:-1]),
            repay_plan=query_repay_plan.query(store_code=str(store_codes)[1:-1]),
            cci_order=query_cci_order.query(store_code=str(store_codes)[1:-1]),
            cci_refund=query_cci_refund.query(store_code=str(store_codes)[1:-1]),
        )

        for store_code in store_codes:
            result = {}
            try:
                if args.traceback:
                    today = all_store_traceback_date.get(store_code, today)

                data = get_store_data(batch_data, store_code, today.strftime("%Y-%m-%d"))
                query_store_data = all_store_data.set_index("C_STORE_CODE").loc[store_code]
                # 编号
                result["C_STORE_CODE"] = store_code
                # 商户名称
                result["C_STORE_NAME"] = query_store_data["C_STORE_NAME"]
                # 单位员工数
                result["GEXshvar009"] = query_store_data["N_STAFF_NUM"]
                # 当前商户评级（盈利类）
                result["GEXshvar010"] = query_store_data["C_GRADE_STOCK_PROFIT"]
                # 当前商户评级（逾期类）
                result["GEXshvar011"] = query_store_data["C_GRADE_STOCK_BAD_LOAN"]
                # 门店经营年限
                result["GEXshvar073"] = relativedelta(query_store_data["D_BUS_END_DATE"], today).years
                # 日均结算笔数
                result["GEXshvar076"] = query_store_data["N_ORDER_NUM_AVG_DAILY_CCI"]
                # 日均结算金额
                result["GEXshvar077"] = query_store_data["N_PAY_AMOUNT_AVG_DAILY_CCI"]
                # 月均结算笔数
                result["GEXshvar081"] = query_store_data["N_ORDER_NUM_AVG_MONTHLY_CCI"]
                # 月均结算金额
                result["GEXshvar082"] = query_store_data["N_PAY_AMOUNT_AVG_MONTHLY_CCI"]
                # 一级行业名称
                result["C_INDUSTRY_ONE"] = query_store_data["C_INDUSTRY_ONE"]
                # 二级行业名称
                result["C_INDUSTRY_TWO"] = query_store_data["C_INDUSTRY_TWO"]
                # 是否B端商户
                result["N_MERCHANT_LOAN_FLAG"] = query_store_data["N_MERCHANT_LOAN_FLAG"]

                try:
                    if args.zjj_status and (data.get("cci_order") is not None and len(data["cci_order"]) > 0) or (
                            data.get("cci_refund") is not None and len(data["cci_refund"]) > 0):
                        feature_zjj_order = deal_store_zjj_order_features(data, today, store_code)
                        result.update({"STORE_ZJJ_FEATURE": 1})
                    else:
                        feature_zjj_order = {}
                        result.update({"STORE_ZJJ_FEATURE": 0})

                    if args.derive_status and (data.get("app_base") is not None and len(data["app_base"]) > 0) or (
                            data.get("loan_base") is not None and len(data["loan_base"]) > 0):
                        feature_derive = deal_store_derive_features(data, today, store_code)
                        result.update({"STORE_DERIVE_FEATURE": 1})
                    else:
                        feature_derive = {}
                        result.update({"STORE_DERIVE_FEATURE": 0})

                    result.update(feature_zjj_order)
                    result.update(feature_derive)
                    result.update({"STORE_ALL_FEATURE_NUM": len(result)})
                    logger.info(f"门店: {store_code}\t\t变量解析个数: {len(result)}")
                    results.append(result)
                except:
                    logger.error(traceback.format_exc())
            except:
                logger.error(traceback.format_exc())
            finally:
                global store_batch_iter
                store_batch_iter.set_postfix(store=store_code, feature=len(result))
                store_batch_iter.update(1)

        return results


    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    parsed_store_code = query_parsed_store.query(table_name=f"{args.database}.{args.table_name}",
                                                 update_time=today.strftime("%Y-%m-%d"))

    zjj_order_feature_info = pd.read_excel("./feature_calculator/商户线上经营监控.xlsx", sheet_name="经营监控变量").drop_duplicates(
        "变量编号", keep="last")
    store_feature_info = pd.read_excel("./feature_calculator/商户线上经营监控.xlsx", sheet_name="商户变量").drop_duplicates("变量名称",
                                                                                                                keep="last")

    if args.debug:
        all_store_code = ['02QYYL01', '02TYNFS01', '02WQP01', 'BJHXP01', 'JYQML01', 'ZZMZM01']
        if parsed_store_code is not None and len(parsed_store_code) > 0:
            all_store_code = list(set(all_store_code) - set(parsed_store_code["C_STORE_CODE"].unique()))
    elif args.traceback:
        all_store_code = list(traceback_store_data["C_STORE_CODE"].unique())
    else:
        if parsed_store_code is not None and len(parsed_store_code) > 0:
            all_store_code = list(all_store_data[~all_store_data["C_STORE_CODE"].isin(parsed_store_code["C_STORE_CODE"])]["C_STORE_CODE"].unique())
        else:
            all_store_code = list(all_store_data["C_STORE_CODE"].unique())


    if args.clear is False and os.path.isfile("parsed_results.pkl"):
        results = load_pickle("parsed_results.pkl")
        parsed_results_code = set(r.get("C_STORE_CODE") for r in results)
        if parsed_results_code is not None and len(parsed_results_code) > 0:
            all_store_code = list(set(all_store_code) - parsed_results_code)

    if len(all_store_code) > 0:
        logger.info(f"解析商户数量 {len(all_store_code)} ......")

        store_batch_iter = tqdm(range(len(all_store_code)), total=len(all_store_code), desc="解析商户变量: ")
        # 多线程运行保证数据查询效率
        if args.max_workers > 1:
            from functools import partial
            from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED, as_completed

            executor = ThreadPoolExecutor(max_workers=args.max_workers)
            parse_feature_func = partial(deal_store_features, all_store_data=all_store_data, today=today)
            parse_feature_tasks = [executor.submit(parse_feature_func, all_store_code[i:i+args.batch_size]) for i in range(0, len(all_store_code), args.batch_size)]

            # wait(parse_feature_tasks, return_when=ALL_COMPLETED)
            # for task in parse_feature_tasks:
            for task in as_completed(parse_feature_tasks):
                if task.result():
                    results.extend(task.result())
        else:
            # 单线程解析商户订单流水相关变量
            for i in range(0, len(all_store_code), args.batch_size):
                result = deal_store_features(all_store_code[i:i + args.batch_size], all_store_data=all_store_data, today=today)
                results.extend(result)

        if len(results) > 0:
            store_features = pd.DataFrame(results)

            # 行业偏离度相关变量
            industry_col = ["GEXshvar054_1", "GEXshvar027", "GEXshvar038", "GEXshvar048", "GEXshvar036", "GEXshvar055",
                            "GEXshvar060", "GEXshvar096", "GEXshvar099", "GEXshvar107", "GEXshvar110", "GEXshvar113",
                            "GEXshvar116", "GEXshvar119", "GEXshvar122", ]
            industry_feature = ["GEXshvar054", "GEXshvar028", "GEXshvar039", "GEXshvar047", "GEXshvar037", "GEXshvar056",
                                "GEXshvar057", "GEXshvar097", "GEXshvar100", "GEXshvar108", "GEXshvar111", "GEXshvar114",
                                "GEXshvar117", "GEXshvar120", "GEXshvar123"]
            industry_zjj_order_features_mean = store_features[industry_col].replace(-1,
                                                                                    np.nan).groupby(
                store_features["C_INDUSTRY_TWO"].fillna("")).mean()
            store_features[industry_feature] = store_features[industry_col] - store_features["C_INDUSTRY_TWO"].fillna(
                "").apply(lambda x: industry_zjj_order_features_mean[industry_col].loc[x])
            # 新增日期相关变量
            store_features["D_UPDATE_DATE"] = today

            # 临时保存数据文件
            save_pickle(store_features, "./store_features.pkl")
            # 处理无穷大值存数据库报错
            store_features = store_features.replace(-np.inf, -args.inf).replace(np.inf, args.inf)

            # 创建存储表
            generate_oracle_create_query(store_features, f"{args.database}.{args.table_name}", if_drop=args.replace)
            # 向表中插入数据
            generate_oracle_insert_query(store_features, f"{args.database}.{args.table_name}")

            if args.replace:
                # 增加数据库表注释
                db_connect_pool.execute(f"COMMENT ON TABLE {args.database}.{args.table_name} IS '商户相关变量'")

                # 增加数据库字段注释
                for _, row in zjj_order_feature_info.iterrows():
                    feature, name = row["变量编号"], row["三级标签"]
                    if feature in store_features.columns and ~pd.isnull(row["三级标签"]) and row["三级标签"] != "":
                        db_connect_pool.execute(f"COMMENT ON COLUMN {args.database}.{args.table_name}.{feature} IS '{name}'")

                for _, row in store_feature_info.iterrows():
                    feature, name = row["变量名称"], row["变量名"]
                    if feature in store_features.columns and ~pd.isnull(row["变量名"]) and row["变量名"] != "":
                        db_connect_pool.execute(f"COMMENT ON COLUMN {args.database}.{args.table_name}.{feature} IS '{name}'")
    else:
        logger.info(f"相关商户变量已全部解析成功")

    zjj_order_feature_info["变量编号"] = zjj_order_feature_info["变量编号"].str.upper()
    store_feature_info["变量名称"] = store_feature_info["变量名称"].str.upper()

    def output_store_feature_attacment(table_name="ZHANGLUPING.DM_MID_STORE_FEATURES"):
        start_row, start_col = 2, 2
        writer = ExcelWriter(style_excel="./tools/报告输出模版.xlsx", theme_color="4256f1")

        worksheet = writer.get_sheet_by_name("收单变量数据字典")
        end_row, end_col = writer.insert_df2sheet(worksheet,
                                                  zjj_order_feature_info[["标签类别", "变量编号", "一级标签", "二级标签", "三级标签"]],
                                                  (start_row, start_col), header=True)

        worksheet = writer.get_sheet_by_name("衍生变量数据字典")
        end_row, end_col = writer.insert_df2sheet(worksheet,
                                                  store_feature_info[["变量分组", "变量名称", "变量名", "数据类型", "变量定义/逻辑"]],
                                                  (start_row, start_col), header=True)

        store_features = query_newest_store_features.query(table_name=table_name)

        if store_features is not None and len(store_features) > 0:
            store_features["D_UPDATE_DATE"] = store_features["D_UPDATE_DATE"].dt.strftime("%Y-%m-%d")
            worksheet = writer.get_sheet_by_name("商户相关变量")
            end_row, end_col = writer.insert_df2sheet(worksheet, store_features, (1, 1), header=True, n_jobs=16)

            writer.save("商户相关变量.xlsx")
        else:
            raise Exception("商户变量数据读取错误")

    logger.info(f"商户变量存excel表 ...")
    output_store_feature_attacment(table_name=f"{args.database}.{args.table_name}")

    # 释放锁
    if os.path.isfile("parse_lock"):
        os.remove("parse_lock")
