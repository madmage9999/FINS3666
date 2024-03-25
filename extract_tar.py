import pandas as pd
import numpy as np
import os
import csv
import plotly.express as px
import plotly.graph_objects as go
import math
import csv
import tarfile
import zipfile
import bz2
import glob
import ast

from datetime import date, timedelta
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)
# Variables
log1_Start = 60 * 30 # Seconds before scheduled off to start recording data for data segment one
log1_Step = 60       # Seconds between log steps for first data segment
log2_Start = 60 * 10  # Seconds before scheduled off to start recording data for data segment two
log2_Step = 1        # Seconds between log steps for second data segment

trading = betfairlightweight.APIClient(username = "username", password = "password", app_key="app_key")
listener = StreamListener(max_latency=None)

# General Utility Functions
# _________________________________

def as_str(v) -> str:
    return '%.2f' % v if type(v) is float else v if type(v) is str else ''

def split_anz_horse_market_name(market_name: str) -> tuple:
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace
    return (race_no, race_len, race_type)


def load_markets(file_paths):
    for file_path in file_paths:
        print(file_path)
        if os.path.isdir(file_path):
            for path in glob.iglob(file_path + '**/**/*.bz2', recursive=True):
                f = bz2.BZ2File(path, 'rb')
                yield f
                f.close()
        elif os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1]
            # iterate through a tar archive
            if ext == '.tar':
                with tarfile.TarFile(file_path) as archive:
                    for file in archive:
                        yield bz2.open(archive.extractfile(file))
            # or a zip archive
            elif ext == '.zip':
                with zipfile.ZipFile(file_path) as archive:
                    for file in archive.namelist():
                        yield bz2.open(archive.open(file))

    return None

def slicePrice(l, n):
    try:
        x = l[n].price
    except:
        x = ""
    return(x)

def sliceSize(l, n):
    try:
        x = l[n].size
    except:
        x = ""
    return(x)

def pull_ladder(availableLadder, n = 5):
        out = {}
        price = []
        volume = []
        if len(availableLadder) == 0:
            return(out)        
        else:
            for rung in availableLadder[0:n]:
                price.append(rung.price)
                volume.append(rung.size)

            out["p"] = price
            out["v"] = volume
            return(out)
        
def pull_ladder_traded(availableLadder, n = 5):
        out = {}
        price = []
        volume = []
        if len(availableLadder) == 0:
            return(out)        
        else:
            for rung in availableLadder:
                price.append(rung.price)
                volume.append(rung.size)

            out["p"] = price
            out["v"] = volume
            return(out)

def filter_market(market: MarketBook) -> bool: 

    d = market.market_definition
    track_filter = ['Bendigo', 'Sandown', 'Flemington', 'Caulfield', 'Moonee Valley']

    return (d.country_code == 'AU' 
        and d.venue in track_filter
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')
 
def final_market_book(s):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        for market_books in gen():

            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++

            if ((evaluate_market := filter_market(market_books[0])) == False):
                    return(None)

            for market_book in market_books:

                last_market_book = market_book

        return(last_market_book)

def parse_final_selection_meta(dir, out_file):

    with open(out_file, "w+") as output:

        output.write("market_id,selection_id,venue,market_time,selection_name,win,bsp\n")

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            last_market_book = final_market_book(stream)

            if last_market_book is None:
                continue 

            # Extract Info ++++++++++++++++++++++++++++++++++

            runnerMeta = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in last_market_book.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'win': np.where(r.status == "WINNER", 1, 0),
                    'sp': r.sp.actual_sp
                }
                for r in last_market_book.runners 
            ]

            # Return Info ++++++++++++++++++++++++++++++++++

            for runnerMeta in runnerMeta:

                if runnerMeta['selection_status'] != 'REMOVED':

                    output.write(
                        "{},{},{},{},{},{},{}\n".format(
                            str(last_market_book.market_id),
                            runnerMeta['selection_id'],
                            last_market_book.market_definition.venue,
                            last_market_book.market_definition.market_time,
                            runnerMeta['selection_name'],
                            runnerMeta['win'],
                            runnerMeta['sp']
                        )
                    )

def loop_preplay_prices(s, o):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        marketID = None
        tradeVols = None
        time = None

        for market_books in gen():

            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++

            if ((evaluate_market := filter_market(market_books[0])) == False):
                    break

            for market_book in market_books:

                # Time Step Management ++++++++++++++++++++++++++++++++++

                if marketID is None:

                    # No market initialised
                    marketID = market_book.market_id
                    time =  market_book.publish_time

                elif market_book.inplay:

                    # Stop once market goes inplay
                    break

                else:

                    seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()

                    if seconds_to_start > log1_Start:

                        # Too early before off to start logging prices
                        continue

                    else:

                        # Update data at different time steps depending on seconds to off
                        wait = np.where(seconds_to_start <= log2_Start, log2_Step, log1_Step)

                        # New Market
                        if market_book.market_id != marketID:
                            marketID = market_book.market_id
                            time =  market_book.publish_time
                        # (wait) seconds elapsed since last write
                        elif (market_book.publish_time - time).total_seconds() > wait:
                            time = market_book.publish_time
                        # fewer than (wait) seconds elapsed continue to next loop
                        else:
                            continue

                # Execute Data Logging ++++++++++++++++++++++++++++++++++
                for runner in market_book.runners:
                    try:
                        atb_ladder = pull_ladder(runner.ex.available_to_back, n = 10)
                        atl_ladder = pull_ladder(runner.ex.available_to_lay, n = 10)
                        traded_volume_ladder = pull_ladder_traded(runner.ex.traded_volume, n = 10)
                    except:
                        atb_ladder = {}
                        atl_ladder = {}
                        traded_volume_ladder = {}

                    # Calculate Current Traded Volume + Tradedd WAP
                    limitTradedVol = sum([rung.size for rung in runner.ex.traded_volume])
                    if limitTradedVol == 0:
                        limitWAP = ""
                    else:
                        limitWAP = sum([rung.size * rung.price for rung in runner.ex.traded_volume]) / limitTradedVol
                        limitWAP = round(limitWAP, 2)

                    o.writerow(
                        (
                            market_book.market_id,
                            runner.selection_id,
                            market_book.publish_time,
                            limitTradedVol,
                            limitWAP,
                            runner.last_price_traded or "",
                            str(atb_ladder).replace(' ',''), 
                            str(atl_ladder).replace(' ',''),
                            str(traded_volume_ladder).replace(' ','')
                        )
                    )

def parse_preplay_prices(dir, out_file):

    with open(out_file, "w+") as output:

        writer = csv.writer(
            output, 
            delimiter=',',
            lineterminator='\r\n',
            quoting=csv.QUOTE_ALL
        )

        writer.writerow(("market_id","selection_id","time","traded_volume","wap","ltp","atb_ladder","atl_ladder","traded_volume_ladder"))

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            loop_preplay_prices(stream, writer)

if __name__ == '__main__':
    month = '2023_12'
    output_dir = f'extracted_data/{month}'
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    # Input file
    stream_files = glob.glob(f"Data2/{month}_ProAUSThoroughbreds.tar") # Change to your YYYY_MM_ProAUSThoroughBeds.tar file

    # Extracting Selection Metadata
    selection_meta = f"{output_dir}/selection_meta.csv"

    print("__ Parsing Selection Metadata ___ ")
    parse_final_selection_meta(stream_files, selection_meta)

    # Extracting Preplay Price Data
    preplay_price_file =  f"{output_dir}/preplay_price.csv"

    print("__ Parsing Detailed Preplay Prices ___ ")
    parse_preplay_prices(stream_files, preplay_price_file)

    print("__ Merging Selection and Price Preplay __")
    selection = pd.read_csv(f"{output_dir}/selection_meta.csv" , dtype={'market_id': object, 'selection_id': object}, parse_dates = ['market_time'])
    prices = pd.read_csv(
            f"{output_dir}/preplay_price_test.csv", 
            quoting=csv.QUOTE_ALL,
            dtype={'market_id': 'string', 'selection_id': 'string', 'atb_ladder': 'string', 'atl_ladder': 'string', 'traded_volume_ladder': 'string'},
            parse_dates=['time']
            )
    
    # To get pandas to correctly recognise the ladder columns as dictionaries
    prices['atb_ladder'] = [ast.literal_eval(x) for x in prices['atb_ladder']]
    prices['atl_ladder'] = [ast.literal_eval(x) for x in prices['atl_ladder']]
    prices['traded_volume_ladder'] = [ast.literal_eval(x) for x in prices['traded_volume_ladder']]

    # Simple join on market and selection_id initially
    df = selection.merge(prices, on = ['market_id', 'selection_id'])
    # Assuming you're using pandas to convert the string to a datetime object
    df['market_time'] = pd.to_datetime(df['market_time'], format='%Y-%m-%d %H:%M:%S')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # Round all datetime objects to the nearest second
    df['time'] = df['time'].dt.round('s')

    df = (
        df
        .assign(back_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atb_ladder']])
        .assign(lay_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atl_ladder']])
        .assign(back_vwap = lambda x: [np.nan if d.get('p') is None else round(sum([a*b for a,b in zip(d.get('p')[0:3],d.get('v')[0:3])]) / sum(d.get('v')[0:3]),3) for d in x['atb_ladder']])
        .assign(lay_vwap = lambda x: [np.nan if d.get('p') is None else round(sum([a*b for a,b in zip(d.get('p')[0:3],d.get('v')[0:3])]) / sum(d.get('v')[0:3]),3) for d in x['atl_ladder']])
        .assign(seconds_before_scheduled_jump = lambda x: round((x['market_time'] - x['time']).dt.total_seconds()))
        .query('seconds_before_scheduled_jump < 1800 and seconds_before_scheduled_jump > -120')
    )

    df.to_csv(f'{output_dir}/{month}_preprocessed.csv', index=False)
