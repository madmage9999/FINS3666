import pandas as pd
import numpy as np
import math
import ast

def trade(input):
    # print(input.atb_ladder)
    input['atb_ladder'] = [ast.literal_eval(x) for x in input['atb_ladder']]
    input['atl_ladder'] = [ast.literal_eval(x) for x in input['atl_ladder']]
    input[['best_BV', 'best_LV']] = input.apply(lambda row: get_price_volume(row), axis=1, result_type='expand')
    input[['expected_price']] = input.apply(lambda row: get_EP(row), axis=1, result_type='expand')
    #input[['lay_BP', 'lay_LP', 'back_BP', 'back_LP']] = input.apply(lambda row: get_spread(row), axis=1, result_type='expand')
    # if liabilityk < 0 prefer to back
    # if 0<liabilityk<limit don't trade
    # if limit < liabilityk prefer to lay
    print(input.values)


def get_price_volume(row):
    best_bv = row.atb_ladder['v'][0]
    best_lv = row.atl_ladder['v'][0]
    return pd.Series([best_bv, best_lv])

def get_EP(row):
    best_BV = row.best_BV
    best_LV = row.best_LV
    best_BP = row.back_best
    best_LP = row.lay_best
    P_up = best_BV/(best_BV+best_LV)
    xprice = (best_BP *(1 - P_up)) + (best_LP * P_up)
    return pd.Series([xprice])

# def get_spread(row):
#     amt_on_market = row.traded_volume
#     total_traded = 
#     traded_prices = 
#     actual_spread = row.best_LV - row.best_BV
#     tick = 
#     xprice = row.expected_price
#     liquidity_ratio = amt_on_market/total_traded * 100
#     stdev = np.std(traded_prices)
#     spread = max(1, round(pow(math.e, -liquidity_ratio)*actual_spread/2*(1+stdev)))
#     return pd.Series([xprice + spread * tick, xprice - ((spread + 1) * tick), xprice + ((spread + 1) * tick), xprice - (spread * tick)])

def get_liability(row, tradebook):
    pass

def get_tick(price):
    if price <= 2:
        return 0.01
    elif price <= 3:
        return 0.02
    elif price <=4:
        return 0.05
    elif price <= 6:
        return 0.1
    elif price <= 10:
        return 0.2
    elif price <= 20:
        return 0.5
    elif price <=30:
        return 1
    elif price <= 50:
        return 2
    elif price <= 100:
        return 5
    else:
        return 10

if __name__ == "__main__":
    chunksize = 8 # get num runners in market
    orderbook = pd.DataFrame()
    tradebook = pd.DataFrame()
    for chunk in pd.read_csv('test_data.csv', chunksize=chunksize):
        trade(chunk)
        break