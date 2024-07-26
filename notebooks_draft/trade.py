import pandas as pd
import numpy as np
import math
import ast

def stream(input, tradebook):
    # print(input.atb_ladder)
    input['atb_ladder'] = [ast.literal_eval(x) for x in input['atb_ladder']]
    input['atl_ladder'] = [ast.literal_eval(x) for x in input['atl_ladder']]
    input['traded_volume_ladder'] = [ast.literal_eval(x) for x in input['traded_volume_ladder']]

    input[['best_BV', 'best_LV']] = input.apply(lambda row: get_price_volume(row), axis=1, result_type='expand')
    input[['expected_price']] = input.apply(lambda row: get_EP(row), axis=1, result_type='expand')
    print(input.expected_price)
    input[['total_volume']] = input['traded_volume'].sum()
    input[['lay_BP', 'lay_LP', 'back_BP', 'back_LP']] = input.apply(lambda row: get_spread(row), axis=1, result_type='expand')
    # print(input)
    input.apply(lambda row: trade(row, tradebook), axis=1)
    

def trade(row, tradebook):
    '''trade/row'''
    row['liability'] = get_liability(row, tradebook)
    #print(row.liability)
    selection = row['selection_id']
    selection = tradebook.loc[tradebook['selection_id'] == selection].iloc[0]
    # if liabilityk < 0 prefer to back
    # if 0<liabilityk<limit don't trade -> trade we want to take on risk so we can make money
    # if limit < liabilityk prefer to lay

    if row.liability.values[0] < 0:
        # print('prefer to back')  # Submit the order
        selection.back_orders['p'].append(row.back_BP)
        selection.back_orders['v'].append(1)
    elif row.liability.values[0] > 1000:
        # if greater than limit lay
        # print('prefer to lay')
        selection.lay_orders['p'].append(row.lay_LP)
        selection.lay_orders['v'].append(1)
    else:
        # print('trade')
        # otherwise trade both sides with largest spread  # Submit the order
        selection.back_orders['p'].append(row.back_BP)
        selection.back_orders['v'].append(1)

        selection.lay_orders['p'].append(row.lay_LP)
        selection.lay_orders['v'].append(1)

    for idx, back in enumerate(selection.back_orders['p']):
        if row.back_best >= back:
            selection.back_trades['p'].append(back)
            selection.back_trades['v'].append(selection.back_orders['v'][idx])
            selection.back_orders['p'].pop(idx)
            selection.back_orders['v'].pop(idx)
            #append to trade book

    for idx, lay in enumerate(selection.lay_orders['p']):
        if row.lay_best <= lay:
            selection.lay_trades['p'].append(lay)
            selection.lay_trades['v'].append(selection.lay_orders['v'][idx])
            selection.lay_orders['p'].pop(idx)
            selection.lay_orders['v'].pop(idx)

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
    back_BP =float('%.2f'%(tick_round(xprice + spread * tick)))
    back_LP =float('%.2f'%(tick_round(xprice - ((spread + 1) * tick))))
    lay_BP = float('%.2f'%(tick_round(xprice + ((spread + 1) * tick))))
    lay_LP = float('%.2f'%(tick_round(xprice - (spread * tick))))
    return pd.Series([back_BP, back_LP, lay_BP, lay_LP])
# Function to calculate total liability on the k-th horse

def get_liability(row, tradebook):
    selection = row.selection_id
    selection = tradebook.loc[tradebook['selection_id'] == selection].iloc[0]
    tradebook['lay_v_sum'] = tradebook['lay_trades'].apply(sum_v_values)
    sum_X = tradebook['lay_v_sum'].sum() # Amount layed on all horses(X_i)
    tradebook['back_v_sum'] = tradebook['back_trades'].apply(sum_v_values)
    sum_Y = sum_X + tradebook['back_v_sum'].sum() # Amount backed and layed on all horses (Y_i)

    Y_k = selection.back_trades['v']  # Amounts betted on each back-order (Y_{k,l}) on the K'th horse
    BP_k = selection.back_trades['p'] # Back prices (BP_{k,l}) on the K'th horse
    X_k = selection.lay_trades['v']  # Amounts betted on each lay-order (X_{k,j}) on the K'th horse
    LP_k = selection.lay_trades['p']  # Lay prices (LP_{k,j}) on the K'th horse

    # Indicator function for the k-th horse win
    # I_k = 1/row.expected_price
    I_k = 1

    #changing ik gives interesting results, keeping it at 1 for now to assume if all horses will win we need to match their liability
    # Total liability calculation
    TL_k = sum_X - sum_Y + I_k * (sum(np.multiply(Y_k, BP_k)) - sum(np.multiply(X_k, LP_k)))
    #TL_K = sum(volume layed) - sum(volume layed and backed)
    # + W or Loss * (sum(if kth horse wins, return from lay and back) - sum(amount payable if kth horse wins))   
    '''
    if horse loses then liablity is how much is backed -b
    else if horse wins then liablity is backed -b +profit back -loss lay
    '''
    # print(TL_k)
    return pd.Series([TL_k])

def sum_p_values(row):
    return sum(row['p'])

def sum_v_values(row):
    return sum(row['v'])

def tick_round(price):
    tick = get_tick(price)
    return tick * round(price/tick)

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