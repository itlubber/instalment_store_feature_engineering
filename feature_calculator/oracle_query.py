from utils import logger, OracleQuery, last_day_of_month


# 查询商户变量数据
query_newest_store_features = OracleQuery(
    """
    select * from {table_name}
    where D_UPDATE_DATE = (select max(D_UPDATE_DATE) from {table_name})
    order by GEXSTORE015
    """
)


def query_traceback_store_data(ofline_file="feature_calculator/门店坏账及盈利含回溯_202304.xlsx", flag=1):
    import numpy as np
    import pandas as pd

    def deal_target(row):
        if pd.isnull(row['坏账率']):
            return np.nan
        elif row['坏账率'] >= 0.2:
            return 1
        else:
            return 0

    df = pd.read_excel(ofline_file)
    df['D_CONCLUSION_TIME'] = pd.to_datetime(df['统计月'],format='%Y%m').apply(last_day_of_month)
    if flag == 1:
        df['target'] = df.apply(lambda row: deal_target(row), axis=1)
        df = df[~df["target"].isnull()].reset_index(drop=True)
        positive = df[df['target'] == 1].sort_values('D_CONCLUSION_TIME', ascending=True).rename(
            columns={"门店编码": "C_STORE_CODE"}).drop_duplicates('C_STORE_CODE', keep='first')
        return positive[['C_STORE_CODE', 'D_CONCLUSION_TIME']]
    if flag == 2:
        positive = OracleQuery(
            """
                SELECT C_STORE_CODE, D_CONCLUSION_TIME FROM datamart.DM_MONITOR_STORE_REPORT 
                WHERE regexp_like(c_conclusion_name,'商户欺诈|中介欺诈|联合客户作案|联合中介作案|套现|传销|工作套路贷|返现|美容套路贷')
                        AND not regexp_like(c_conclusion_name,'代申请|疑似欺诈|虚假术后|未能定性欺诈|白马欺诈')
            """
        ).query()
        positive['D_CONCLUSION_TIME'] = pd.to_datetime(positive['D_CONCLUSION_TIME'])
        positive = positive.sort_values('D_CONCLUSION_TIME', ascending=True).drop_duplicates('C_STORE_CODE', keep='first')
        return positive


# 商户基础信息
query_all_store_data = OracleQuery(
    """
    with store_base as (
        select DISTINCT C_STORE_CODE
                , C_STORE_NAME
                , C_STORE_PROVINCE
                , C_STORE_CITY
                , N_STAFF_NUM
                , D_BUS_END_DATE
                , C_STORE_LEVEL
                , C_STORE_LEVEL_TYPE
                , C_GRADE_STOCK_PROFIT
                , C_GRADE_STOCK_BAD_LOAN
                , C_INDUSTRY_ONE
                , C_INDUSTRY_TWO
                , N_MERCHANT_LOAN_FLAG
            from DATAMART.DM_MID_STORE_ORGNZ
            -- where C_STORE_CODE in (
            --     select C_STORE_CODE from DATAMART.DM_ZJ_CCI_ORDER 
            --     -- WHERE C_PAY_STATUS = 'S' 
            --     group by C_STORE_CODE
            -- ) 
    ), loan_base_daily AS (
        SELECT C_STORE_CODE, AVG(N_EXT_ID_NUM) N_EXT_ID_NUM_AVG_DAILY, AVG(N_LOAN_AMOUNT) N_LOAN_AMOUNT_AVG_DAILY
        FROM (
            select C_STORE_CODE ,TO_CHAR(D_LOAN_DATE, 'yyyy-mm-dd') D_LOAN_DATE, SUM(N_LOAN_AMOUNT) N_LOAN_AMOUNT, COUNT(C_EXT_ID) N_EXT_ID_NUM
            from DATAMART.DM_MID_LOAN_BASE dmlb 
            WHERE C_FINANCE_PRODUCT_CODE NOT LIKE 'DDG%'
            GROUP BY C_STORE_CODE ,TO_CHAR(D_LOAN_DATE, 'yyyy-mm-dd')
        ) t
        GROUP BY C_STORE_CODE
    ), loan_base_monthly AS (
        SELECT C_STORE_CODE, AVG(N_EXT_ID_NUM) N_EXT_ID_NUM_AVG_MONTHLY, AVG(N_LOAN_AMOUNT) N_LOAN_AMOUNT_AVG_MONTHLY
        FROM (
            select C_STORE_CODE ,TO_CHAR(D_LOAN_DATE, 'yyyy-mm') D_LOAN_DATE, SUM(N_LOAN_AMOUNT) N_LOAN_AMOUNT, COUNT(C_EXT_ID) N_EXT_ID_NUM
            from DATAMART.DM_MID_LOAN_BASE dmlb 
            WHERE C_FINANCE_PRODUCT_CODE NOT LIKE 'DDG%'
            GROUP BY C_STORE_CODE ,TO_CHAR(D_LOAN_DATE, 'yyyy-mm')
        ) t
        GROUP BY C_STORE_CODE
    ), cci_order_daily AS (
        SELECT C_STORE_CODE, AVG(N_ORDER_NUM) N_ORDER_NUM_AVG_DAILY_CCI, AVG(N_PAY_AMOUNT)  N_PAY_AMOUNT_AVG_DAILY_CCI
        FROM (
            SELECT C_STORE_CODE, TO_CHAR(D_CREATE_TIME, 'yyyy-mm-dd') D_CREATE_TIME ,COUNT(C_ORDER_NO) N_ORDER_NUM,SUM(N_PAY_AMOUNT) N_PAY_AMOUNT, COUNT(DISTINCT C_USER_ID) N_USER_NUM
            FROM DATAMART.DM_ZJ_CCI_ORDER
            WHERE C_PAY_STATUS = 'S'
                    -- AND (C_RES_MSG LIKE '%成功%' OR  C_RES_MSG LIKE '%OK%')
            GROUP BY C_STORE_CODE, TO_CHAR(D_CREATE_TIME, 'yyyy-mm-dd') 
        ) t
        GROUP BY C_STORE_CODE
    ), cci_order_monthly AS (
        SELECT C_STORE_CODE, AVG(N_ORDER_NUM) N_ORDER_NUM_AVG_MONTHLY_CCI, AVG(N_PAY_AMOUNT)  N_PAY_AMOUNT_AVG_MONTHLY_CCI
        FROM (
            SELECT C_STORE_CODE, TO_CHAR(D_CREATE_TIME, 'yyyy-mm') D_CREATE_TIME ,COUNT(C_ORDER_NO) N_ORDER_NUM,SUM(N_PAY_AMOUNT) N_PAY_AMOUNT, COUNT(DISTINCT C_USER_ID) N_USER_NUM
            FROM DATAMART.DM_ZJ_CCI_ORDER
            WHERE C_PAY_STATUS = 'S'
                    -- AND (C_RES_MSG LIKE '%成功%' OR  C_RES_MSG LIKE '%OK%')
            GROUP BY C_STORE_CODE, TO_CHAR(D_CREATE_TIME, 'yyyy-mm') 
        ) t
        GROUP BY C_STORE_CODE
    )
    select store_base.*
            , loan_base_daily.N_EXT_ID_NUM_AVG_DAILY
            , loan_base_daily.N_LOAN_AMOUNT_AVG_DAILY
            , loan_base_monthly.N_EXT_ID_NUM_AVG_MONTHLY
            , loan_base_monthly.N_LOAN_AMOUNT_AVG_MONTHLY
            , cci_order_daily.N_PAY_AMOUNT_AVG_DAILY_CCI
            , cci_order_daily.N_ORDER_NUM_AVG_DAILY_CCI
            , cci_order_monthly.N_PAY_AMOUNT_AVG_MONTHLY_CCI
            , cci_order_monthly.N_ORDER_NUM_AVG_MONTHLY_CCI
    from store_base
    left join loan_base_daily on loan_base_daily.C_STORE_CODE = store_base.C_STORE_CODE
    left join loan_base_monthly on loan_base_monthly.C_STORE_CODE = store_base.C_STORE_CODE
    left join cci_order_daily on cci_order_daily.C_STORE_CODE = store_base.C_STORE_CODE
    left join cci_order_monthly on cci_order_monthly.C_STORE_CODE = store_base.C_STORE_CODE
    -- order by N_LOAN_AMOUNT_AVG_MONTHLY desc
    """
)


query_parsed_store = OracleQuery(
    """
    SELECT DISTINCT C_STORE_CODE
    FROM {table_name}
    WHERE D_UPDATE_DATE = to_date('{update_time}', 'yyyy-mm-dd')
    """
)


# 查询商户的所有放款订单信息
query_loan_base = OracleQuery(
    """
    SELECT *
    FROM DATAMART.DM_MID_LOAN_BASE
    WHERE C_STORE_CODE IN ({store_code}) 
            AND C_FINANCE_PRODUCT_CODE != '1'
    """
)


# 查询商户的所有放宽订单客户的还款计划表
query_repay_plan = OracleQuery(
    """
    SELECT *
    FROM DATAMART.DM_MID_REPAY_PLAN
    WHERE C_EXT_ID in (
        SELECT C_EXT_ID
        FROM DATAMART.DM_MID_LOAN_BASE
        WHERE C_STORE_CODE IN ({store_code})
                AND C_FINANCE_PRODUCT_CODE != '1'
        GROUP BY C_EXT_ID
    )
    """
)


# 查询商户的所有申请订单
query_app_base = OracleQuery(
    """
    SELECT *
    FROM DATAMART.DM_MID_APP_BASE
    WHERE C_STORE_CODE IN ({store_code})
            -- AND N_APP_STATUS IN (130, 140, 160)
            AND C_FINANCE_PRODUCT_CODE != '1'
    """
)


# 查询商户的所有申请订单的评分
query_score_data = OracleQuery(
    """
    SELECT s.C_APP_ID, a.C_CUST_IDNO, a.D_APPLICATION, a.C_STORE_CODE, s.D_CREATE_TIME, s.X517, s.X475, s.X509, s.X506, s.X612, s.X522, s.X631, s.X492
    FROM (
            SELECT C_APP_ID, D_CREATE_TIME, X517, X475, X509, X506, X612, X522, X631, X492, ROW_NUMBER() OVER(partition by C_APP_ID order by D_CREATE_TIME desc) rk
            FROM DATAMART.DM_MID_URULE_PARAM
    ) s
    LEFT JOIN DATAMART.DM_MID_APP_BASE a
    ON s.C_APP_ID = a.C_APP_ID
    WHERE a.C_INDUSTRY_TWO in ('祛痘','整形','口腔','皮肤科','植发','职培')
            AND a.C_STORE_CODE IN ({store_code})
            AND s.rk = 1
            AND (a.C_FINANCE_PRODUCT_TYPE like '无卡%' or a.C_FINANCE_PRODUCT_TYPE like '超G%')
            AND a.C_FINANCE_PRODUCT_CODE != '1'
            AND a.N_APP_STATUS IN (130, 140, 160)
    """
)


# 查询商户上传小票信息
query_bill_upload_data = OracleQuery(
    """
    SELECT c_store_code, C_ORDER_ID, C_ID_NO, D_DRAWDOWN_TIME, D_APPLY_TIME
    FROM dataware.dw_loaning_order
    WHERE c_store_code in ({store_code})
            AND C_FINANCE_PRODUCT_CODE != '1'
            AND C_BILL_UPLOAD IN ('4', '5')
            AND c_order_id in (
                SELECT c_app_id
                FROM DATAMART.DM_MID_APP_BASE
                WHERE c_store_code in ({store_code})
                        AND n_app_status IN (130, 140, 160)
            )
    """
)


# 商户冻结相关信息
query_store_freeze = OracleQuery(
    """
        select C_STORE_CODE, D_BATCH_DATE
        from dataware.dw_mm_freeze_reason
        where c_store_code in ({store_code})
    """
)


# 订单冻结相关信息
query_order_freeze = OracleQuery(
    """
    SELECT C_APP_ID, C_STORE_CODE, D_APPLICATION, D_FREEZE_TIME, D_UNFREEZE_TIME, C_FIRST_FREEZE_REASON 
    FROM DATAMART.DM_ORD_FREEZE_REASON
    WHERE C_STORE_CODE in ({store_code})
            AND C_FINANCE_PRODUCT_ONE != '1'
    """
)


# 商户投诉相关信息
query_store_complaint = OracleQuery(
    """
        select a.c_store_code, c.D_MERCHANT_CODE, c.C_APP_ID, c.D_CREATE, c.c_complaint_type
        from dataware.dw_complaint_base_info c
        left join DATAMART.dm_mid_app_base a on c.C_APP_ID = a.C_APP_ID
        where a.C_STORE_CODE IN ({store_code}) 
                AND (a.C_FINANCE_PRODUCT_TYPE like '无卡%' or a.C_FINANCE_PRODUCT_TYPE like '超G%')
                AND c.C_APP_ID != '\'
                AND REGEXP_LIKE(c.c_complaint_type, '商户欺诈|商户纠纷|学生贷')
    """
)


# 催收情况
query_collection_data = OracleQuery(
    """
    SELECT C_APP_ID, C_STORE_CODE, N_LOSING_FLG, N_CASHOUT_FLG, D_INCOLLECTION_DATE, N_STATUS, N_CURR_PAY_AMOUNT, N_CORPUS_BALANCE, N_ALL_PAY_AMOUNT
    from DATAWARE.DW_INCOLLECTION_POOL 
    WHERE C_STORE_CODE IN ({store_code}) 
            AND C_FINANCE_PRODUCT_CODE != '1'
            AND (N_CASHOUT_FLG in (12,15) or N_LOSING_FLG = 1)
    -- SELECT l.C_STORE_CODE,
    -- c.C_APP_ID, 
    -- max(c.D_ACTION_TIME) D_ACTION_TIME_MAX, 
    -- LISTAGG(c.C_MEMO, ',') WITHIN GROUP(ORDER BY c.C_APP_ID) C_MEMO 
    -- FROM dataware.dw_collection_action c
    -- left join DATAMART.DM_MID_LOAN_BASE l on c.C_APP_ID = l.C_EXT_ID
    -- WHERE c.c_action_code = 'OOC' and l.c_store_code in ({store_code}) 
    -- GROUP BY l.C_STORE_CODE, c.C_APP_ID
    """
)


# 查询商铺基础信息
query_store_info = OracleQuery(
    """
    SELECT *
    FROM DATAMART.DM_MID_STORE_ORGNZ
    WHERE C_STORE_CODE IN ({store_code})
    """
)


# 查询即收款订单流水
query_cci_order = OracleQuery(
    """
    SELECT *
    FROM DATAMART.DM_ZJ_CCI_ORDER
    WHERE C_STORE_CODE IN ({store_code})
            -- AND (C_RES_MSG LIKE '%成功%' OR  C_RES_MSG LIKE '%OK%') 
            AND C_PAY_STATUS = 'S'
    """
)


# 查询即收款订单退货信息
query_cci_refund = OracleQuery(
    """
    SELECT *
    FROM DATAWARE.DW_ZJ_CCI_REFUND
    WHERE C_STORE_CODE IN ({store_code})
            AND C_STATUS = 'S'
    """
)


# 订单案件详情信息
query_order_report = OracleQuery(
    """
    SELECT C_APP_ID, N_APP_STATUS, C_STORE_CODE, D_REPORT_TIME, D_END_TIME, C_CASE_TYPE, C_SURVEY_TYPE, C_SURVEY_DETAIL
    FROM DATAMART.DM_MONITOR_ORDER_REPORT
    WHERE C_STORE_CODE IN ({store_code})
            AND C_FINANCE_PRODUCT_ONE != '1'
            AND REGEXP_LIKE(C_SURVEY_DETAIL, '商户欺诈|中介欺诈|联合客户作案|联合中介作案|套现|传销|工作套路贷|返现|美容套路贷')
    """
)
