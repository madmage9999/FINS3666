import threading
import subprocess
import pandas as pd
import time

# instruction: python run_multiple_trades.py

def run_script(script_name, month):
    # Split the script name and its arguments into separate elements
    subprocess.run(["python", script_name, "-m", month])

def all_summary():
    dec = '2023_12'
    jan = '2024_01'
    feb = '2024_02'

    df1 = pd.read_csv(f'trade_result_3/{dec}/{dec}_summary.csv',
                    dtype={'market_id': 'string', 'selection_id': 'string'},
                    )
    df2 = pd.read_csv(f'trade_result_3/{jan}/{jan}_summary.csv',
                    dtype={'market_id': 'string', 'selection_id': 'string'},
                    )
    df3 = pd.read_csv(f'trade_result_3/{feb}/{feb}_summary.csv',
                    dtype={'market_id': 'string', 'selection_id': 'string'},
                    )
    summary = pd.concat([df1, df2, df3], axis=0)
    summary['gpl'] = summary['back_return'] + summary['lay_return']
    summary['stake'] = summary['back_v_sum'] + summary['lay_v_sum']

    output_dir = f'trade_result_3'
    summary.to_csv(f'{output_dir}/all_months_summary.csv', index=False)

    eval_df = bet_eval_metrics(summary)
    print(f'Total Gross Profit/Loss % = {(eval_df["gpl"]/eval_df["stake"]*100).values[0]}')

def bet_eval_metrics(d):

    metrics = pd.DataFrame(d
    .agg({"gpl": "sum", "stake": "sum"})
    ).transpose().assign(pot=lambda x: x['gpl'] / x['stake'])

    return(metrics[metrics['stake'] != 0])

if __name__ == "__main__":

    # extract1_thread = threading.Thread(target=run_script, args=("./extract_tar.py", "2023_12"))
    # extract2_thread = threading.Thread(target=run_script, args=("./extract_tar.py", "2024_01"))
    # extract3_thread = threading.Thread(target=run_script, args=("./extract_tar.py", "2024_02"))

    # print("Start extraction.")

    # extract1_thread.start()
    # extract2_thread.start()
    # extract3_thread.start()

    # extract1_thread.join()
    # extract2_thread.join()
    # extract3_thread.join()

    # print("All extraction scripts have finished executing.")
    # time.sleep(3)

    trade1_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2023_12"))
    trade2_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2024_01"))
    trade3_thread = threading.Thread(target=run_script, args=("./monthly_trade.py", "2024_02"))

    print("Start backtest.")

    trade1_thread.start()
    trade2_thread.start()
    trade3_thread.start()

    trade1_thread.join()
    trade2_thread.join()
    trade3_thread.join()

    print("All backtest scripts have finished executing.")
    time.sleep(3)

    print("Combining summary.")
    all_summary()

