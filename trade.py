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
    print(input)


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

def get_spread(amt_on_market, total_traded, traded_prices, actual_spread, tick, xprice):
    liquidity_ratio = amt_on_market/total_traded * 100
    stdev = np.std(traded_prices)
    spread = max(1, round(pow(math.e, -liquidity_ratio)*actual_spread/2*(1+stdev)))
    return {
        "lay_BP": xprice + spread * tick,
        "lay_LP": xprice - ((spread + 1) * tick),
        "back_BP": xprice + ((spread + 1) * tick),
        "back_LP": xprice - (spread * tick),
    }

if __name__ == "__main__":
    chunksize = 8
    for chunk in pd.read_csv('test_data.csv', chunksize=chunksize):
        trade(chunk)
        break