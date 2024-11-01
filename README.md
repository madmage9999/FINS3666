﻿# Market Making on Betfair Horse Racing

## Project structure
The project has the following structure:

```
├── Data
│   ├── .bz2 sample file (raw input to extract_tar.py)
├── notebooks_draft
│   ├── jupyter notebooks and python script consisting of drafts while building the trading strategies
├── extract_tar.py 
├── monthly_trade.py 
├── run_multiple_trades.py 
├── result_plotting.ipynb
 ```

`extract_tar.py`: preprocessing raw input .bz2 files to csv file (sourced from Betfair documentation)

`monthly_trade.py`: main script for running trades monthly, includes data streaming, model calculation, and return calculation

`run_multiple_trades.py`: script to run parallel monthly trade scripts using threading and subprocess

`result_plotting.ipynb`: notebook to visualize result and conduct analysis

## Key points
### Spread
Calculating Expected mid price using best bid and ask prices and probability of an upward movement

### Liability
Calculating potential amount that the bookmaker could win or lose if a specific horse wins the race

### Risk management
#### Trading behavior:
- Negative liability < 0 = back more on that specific horse and lay more on all other horses
- 0 < Liability < limit = trade
- Positive liability > limit = lay more on winning horse and back more on all other horses
- Do not trade for least favorite horse

### Inventory management and order sizing
- Adjustment on back and lay price depending on the trading behavior
- Order sizing mechanism following bettors behavior

### Backtesting
We implement our market making model to horse racing data from Dec 2023 - Feb 2024 (3 months)
- Each race includes data from 30 minutes before race starts

