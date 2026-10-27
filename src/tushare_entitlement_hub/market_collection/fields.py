"""Declared provider fields for public market collection; no model or training logic."""

EXTRA_FIELDS = {
    "moneyflow": "ts_code,trade_date,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount",
    "stock_st": "ts_code,name,trade_date,type,type_name",
    "cyq_perf": "ts_code,trade_date,cost_5pct,cost_50pct,cost_95pct,weight_avg",
    "forecast_vip": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,first_ann_date",
    "index_classify": "index_code,industry_name,level,industry_code,is_pub,parent_code,src",
    "index_member_all": (
        "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
    ),
    "sw_daily": "ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount",
    "index_daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "dividend": (
        "ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,"
        "record_date,ex_date,pay_date,div_listdate"
    ),
}
