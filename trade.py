import pandas as pd
import numpy as np
import math
import ast

def stream(input, tradebook):
    # print(input.atb_ladder)
    input['atb_ladder'] = [ast.literal_eval(x) for x in input['atb_ladder']]
    input['atl_ladder'] = [ast.literal_eval(x) for x in input['atl_ladder']]
    input['traded_volume_ladder'] = [ast.literal_eval(x) for x in input['traded_volume_ladder']]
    # tradebook['back_orders'] = [ast.literal_eval(x) for x in tradebook['back_orders']]
    # tradebook['lay_orders'] = [ast.literal_eval(x) for x in tradebook['lay_orders']]
    # tradebook['back_trades'] = [ast.literal_eval(x) for x in tradebook['back_trades']]
    # tradebook['lay_trades'] = [ast.literal_eval(x) for x in tradebook['lay_trades']]
    input[['best_BV', 'best_LV']] = input.apply(lambda row: get_price_volume(row), axis=1, result_type='expand')
    input[['expected_price']] = input.apply(lambda row: get_EP(row), axis=1, result_type='expand')
    input[['total_volume']] = input['traded_volume'].sum()
    input[['lay_BP', 'lay_LP', 'back_BP', 'back_LP']] = input.apply(lambda row: get_spread(row), axis=1, result_type='expand')
    
    input.apply(lambda row: trade(row, tradebook), axis=1)
    
    # if liabilityk < 0 prefer to back
    # if 0<liabilityk<limit don't trade
    # if limit < liabilityk prefer to lay

def trade(row, tradebook):
    '''trade/row'''
    selection = row['selection_id']
    selection = tradebook.loc[tradebook['selection_id'] == selection].iloc[0]

    selection.back_orders['p'].append(row.lay_BP)  # Modify the order
    selection.back_orders['p'].append(row.back_BP)
    selection.back_orders['v'].append(1)
    selection.back_orders['v'].append(1)

    selection.lay_orders['p'].append(row.lay_LP)
    selection.lay_orders['p'].append(row.back_LP)
    selection.lay_orders['v'].append(1)
    selection.lay_orders['v'].append(1)

    for idx, back in enumerate(selection.back_orders['p']):
        if row.back_best >= back:
            selection.back_trades['p'].append(back)
            selection.back_trades['v'].append(selection.back_orders['v'][idx])
            selection.back_orders['p'].pop(idx)
            selection.back_orders['p'].pop(idx)
            #append to trade book

    for idx, lay in enumerate(selection.lay_orders['p']):
        if row.lay_best <= lay:
            selection.lay_trades['p'].append(lay)
            selection.lay_trades['v'].append(selection.lay_orders['v'][idx])
            selection.lay_orders['p'].pop(idx)
            selection.lay_orders['p'].pop(idx)

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

def get_spread(row):
    amt_on_market = row.traded_volume
    total_traded = row.total_volume
    traded_prices = row.traded_volume_ladder['p']
    actual_spread = row.best_LV - row.best_BV
    tick = get_tick((row.back_best+row.lay_best)/2)
    xprice = row.expected_price
    liquidity_ratio = amt_on_market/total_traded * 100
    stdev = np.std(traded_prices)
    spread = max(1, round(pow(math.e, -liquidity_ratio)*actual_spread/2*(1+stdev)))
    spread = (spread + 1)/2
    return pd.Series([xprice + spread * tick, xprice - ((spread + 1) * tick), xprice + ((spread + 1) * tick), xprice - (spread * tick)])
# Function to calculate total liability on the k-th horse

def get_liability(row, tradebook):
    X = np.array([100, 150, 200, 250, 300])  # Amount layed on all horses(X_i)
    Y = np.array([50, 30, 20, 70, 80])  # Amount backed and layed on all horses (Y_i)
    Y_k = np.array([20, 30, 50])  # Amounts betted on each back-order (Y_{k,l})
    BP_k = np.array([2.5, 3.0, 4.0])  # Back prices (BP_{k,l})
    X_k = np.array([10, 15])  # Amounts betted on each lay-order (X_{k,j})
    LP_k = np.array([5.0, 6.0])  # Lay prices (LP_{k,j})

    # Indicator function for the k-th horse win
    I_k = 1/row.expected_price

    # Total liability calculation
    TL_k = sum(X) - sum(Y) + I_k * (sum(Y_k * BP_k) - sum(X_k * LP_k))    
    '''
    if horse loses then liablity is how much is backed -b
    else if horse wins then liablity is backed -b +profit back -loss lay
    '''
    return TL_k

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
    
def init_tradebook():
    df = pd.read_csv('test_data.csv')
    runner_id = df['selection_id'].unique().tolist()
    runner_name = df['selection_name'].unique().tolist()

    # Initialize the tradebook DataFrame
    tradebook = pd.DataFrame({
        'selection_id': runner_id,
        'selection_name': runner_name,
        'back_orders': None,
        'lay_orders': None,
        'back_trades': None,
        'lay_trades': None
    })
    for col in ['back_orders', 'lay_orders', 'back_trades', 'lay_trades']:
        tradebook[col] = tradebook[col].apply(lambda x: {'p': [], 'v': []})
    return tradebook

if __name__ == "__main__":
    # initialise trade book
    tradebook = init_tradebook()
    chunksize = len(tradebook['selection_id'].unique().tolist()) # get num runners in market

    # start trading for each tick in the market stream
    for chunk in pd.read_csv('test_data.csv', chunksize=chunksize):
        stream(chunk, tradebook)
        # print(chunk)
        # print(tradebook)
        # break
    tradebook = tradebook.drop('back_orders', axis=1)
    tradebook = tradebook.drop('lay_orders', axis=1)
    tradebook.to_csv('test_tradebook.csv')