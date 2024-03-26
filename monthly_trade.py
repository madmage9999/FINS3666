import pandas as pd
import numpy as np
import os
import math
import ast

import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--month")
args = parser.parse_args()

def get_price_volume(row):
    best_bv = row.atb_ladder['v'][0] if row.atb_ladder else 0
    best_lv = row.atl_ladder['v'][0] if row.atl_ladder else 0
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
    # for horses we want to back more
    back_BP = float('%.2f'%(tick_round(xprice + spread * tick))) # back price (7)
    back_LP = float('%.2f'%(tick_round(xprice - ((spread + 1) * tick)))) # lay price (8)
    # for horses we want to lay more
    lay_BP = float('%.2f'%(tick_round(xprice + ((spread + 1) * tick)))) # back price (9)
    lay_LP = float('%.2f'%(tick_round(xprice - (spread * tick)))) # lay price (10)
    
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
    
def init_tradebook(race_df):
    
    # Initialize the tradebook DataFrame
    tradebook = race_df[['market_id', 'selection_id', 'selection_name', 'venue', 'win']].drop_duplicates()
    tradebook['back_orders'] = None
    tradebook['lay_orders'] = None
    tradebook['back_trades'] = None
    tradebook['lay_trades'] = None

    for col in ['back_orders', 'lay_orders', 'back_trades', 'lay_trades']:
        tradebook[col] = tradebook[col].apply(lambda x: {'p': [], 'v': []})

    return tradebook

def bet_apply_commission(df, com = 0.05):

    # Total Market GPL
    df['market_gpl'] = df.groupby('market_id')['gpl'].transform(sum)

    # Apply 5% commission
    df['market_commission'] = np.where(df['market_gpl'] <= 0, 0, 0.05 * df['market_gpl'])

    # Sum of Market Winning Bets
    df['floored_gpl'] = np.where(df['gpl'] <= 0, 0, df['gpl'])
    df['market_netwinnings'] = df.groupby('market_id')['floored_gpl'].transform(sum)

    # Partition Commission According to Selection GPL
    df['commission'] = np.where(df['market_netwinnings'] == 0, 0, (df['market_commission'] * df['floored_gpl']) / (df['market_netwinnings']))

    # Calculate Selection NPL
    df['npl'] = df['gpl'] - df['commission']

    # Drop excess columns
    df = df.drop(columns = ['floored_gpl', 'market_netwinnings', 'market_commission', 'market_gpl'])

    return(df)

def stream(input, tradebook):

    try:
        input['atb_ladder'] = [ast.literal_eval(x) for x in input['atb_ladder']]
        input['atl_ladder'] = [ast.literal_eval(x) for x in input['atl_ladder']]
        input['traded_volume_ladder'] = [ast.literal_eval(x) for x in input['traded_volume_ladder']]

        input[['best_BV', 'best_LV']] = input.apply(lambda row: get_price_volume(row), axis=1, result_type='expand')
        input[['expected_price']] = input.apply(lambda row: get_EP(row), axis=1, result_type='expand')
        input[['total_volume']] = input['traded_volume'].sum()
        input[['lay_BP', 'lay_LP', 'back_BP', 'back_LP']] = input.apply(lambda row: get_spread(row), axis=1, result_type='expand')
        
        input['favorites'] = input['bsp'].rank().astype(int)
        # print(np.sum(1/input['expected_price'].to_numpy()))
        capital = 1
        input.apply(lambda row: trade(row, tradebook, capital), axis=1)
    except Exception as e:
        print(e) 
    # limit:= multiplying the overround with the wagered amount

def trade(row, tradebook, capital):
    '''trade/row'''
    try:
        n_horses = len(tradebook)
        # do not trade for least favorite horse
        if row.favorites == n_horses:
            return
        # skip trade for row with incomplete data
        if (not row.atb_ladder) or (not row.atl_ladder) or (not row.traded_volume_ladder):
            print(f"not trading: {row.selection_id} @ market_id: {row.market_id}")
            return 

        row['liability'] = get_liability(row, tradebook)

        selection = row['selection_id']
        selection = tradebook.loc[tradebook['selection_id'] == selection].iloc[0]

        # table 18 model behaviour
        # liability_k = get_liability(_____, tradebook) # not sure 

        # overround = np.sum(1/input['expected_price'].to_numpy()) - 1 # should be the implied probability, not sure to use expected price or bsp?
        # ^ probably use XP, bsp is unknown until the start of the race
        # limit = overround * X # wagered amount, not sure how to get this

        # behaviour = 'backing' if liability_k < 0 else ('laying' if liability_k > limit else 'do not trade')
        
        # ORDER SIZING
        # top 3 favorite horses get 75% of the capital
        if row.favorites <= 3:
            proportion = 0.75 / 3
        else:
            # rest: allocate the remaining 25% with decreasing sequence
            remaining_horses = n_horses - 3
            proportion = (0.25 / remaining_horses) * (n_horses - row.favorites)
        stake = capital * proportion

        if row.liability.values[0] < 0:
            selection.back_orders['p'].append(row.back_BP)
            selection.back_orders['v'].append(stake)
        elif row.liability.values[0] > 1000:
            # if greater than limit lay
            selection.lay_orders['p'].append(row.lay_LP)
            selection.lay_orders['v'].append(stake)
        else:
            # otherwise trade both sides with largest spread  # Submit the order
            selection.back_orders['p'].append(row.back_BP)
            selection.back_orders['v'].append(stake)

            selection.lay_orders['p'].append(row.lay_LP)
            selection.lay_orders['v'].append(stake)

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

    except Exception as e:
            print(e, row)

def yield_chunks(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def run(month: str, ouutput_dir: str):
    
    monthly_data = pd.read_csv(f'extracted_data/{month}/{month}_preprocessed.csv')
    race_groups = monthly_data.groupby('market_id')

    monthly_tradebook = pd.DataFrame()

    for market_id, race_df in race_groups:
        # print(f'Trading for market_id: {market_id}')
        tradebook = init_tradebook(race_df)
        chunksize = len(tradebook['selection_id'].unique().tolist()) # get num runners in market

        # start trading for each tick in the market stream
        for chunk in yield_chunks(race_df, chunksize):
            try:
                stream(chunk, tradebook)
            except Exception as e:
                print(f"Error at market_id: {market_id}, chunk: {chunk.selection_id}, {chunk.seconds_before_scheduled_jump}")
            # break
        tradebook = tradebook.drop('back_orders', axis=1)
        tradebook = tradebook.drop('lay_orders', axis=1)
        monthly_tradebook = pd.concat([monthly_tradebook, tradebook], ignore_index=True)

    monthly_tradebook.to_csv(f'{output_dir}/{month}_tradebook.csv', index=False)

    return monthly_tradebook

def calculate_back_lay_profit_liability(df, output_dir: str):
    '''
    if horse loses then liablity is how much is backed -b + lay profit
    else if horse wins then liablity is +back profit -loss lay
    '''

    #changed label from profit to return because we are summing that column

    df['back_liability'] = 0
    df['back_return'] = 0
    df['lay_liability'] = 0
    df['lay_return'] = 0
    
    for index, row in df.iterrows():
        back_prices = np.array(row['back_trades']['p'])
        back_volumes = np.array(row['back_trades']['v'])
        
        lay_prices = np.array(row['lay_trades']['p'])
        lay_volumes = np.array(row['lay_trades']['v'])
        
        back_liability = sum(back_volumes)
        if row['win'] == 1:
            back_return = np.dot(back_prices, back_volumes) - sum(back_volumes)
            #profit does not include outlay, ie. $1@2.00 = $2 return - $1 stake => $1 profit
        else:
            back_return = -back_liability # lose your backing stake
        
        lay_return = sum(lay_volumes)
        if row['win'] == 1:
            lay_liability = np.dot(lay_volumes, (lay_prices - 1))
            lay_return = -lay_liability
        else:
            lay_liability = 0 # if horse loses then no liablity
        
        df.at[index, 'back_liability'] = back_liability
        df.at[index, 'back_return'] = back_return
        df.at[index, 'lay_liability'] = lay_liability
        df.at[index, 'lay_return'] = lay_return
    
    df.to_csv(f'{output_dir}/{month}_summary.csv', index=False)

    return df

def calculate_return(month, summary):

    total_return = (summary['back_return'] + summary['lay_return']).sum()
    total_trades = (summary['back_v_sum'] + summary['lay_v_sum']).sum()
    return_pct = total_return/total_trades * 100

    # Create a dictionary with the calculated values
    result_dict = {
        'month': [month],
        'total_return': [total_return],
        'total_trades': [total_trades],
        'return_pct': [return_pct]
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(f'{output_dir}/{month}_return.csv', index=False)

    return df


if __name__ == '__main__':
    # initialise trade book
    # month = '2023_12'
    month = args.month
    output_dir = f'trade_result_all_tracks/{month}'
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    print(f"___ Started trading for {month} ___")

    monthly_tradebook = run(month, output_dir)
    summary = calculate_back_lay_profit_liability(monthly_tradebook, output_dir)
    monthly_return = calculate_return(month, summary)

    print(f"___ Finished trading for {month} ___")
    