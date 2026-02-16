"""
Backtesting Script for 21/51 EMA Crossover Strategy with 51 EMA Trailing Stop Loss

Strategy Rules:
1. Entry: 21 EMA crosses 51 EMA (bullish/bearish)
2. Stop Loss: Trails with 51 EMA (dynamic, updates every candle)
3. Take Profit: NONE - ride the trend until SL is hit
4. Position Sizing: Fixed risk per trade ($5)
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# ================= CONFIG =================
SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_M5

# Number of candles to backtest (M5 candles)
# 75,000 candles ≈ 260 days of M5 data
# 50,000 candles ≈ 173 days
# 25,000 candles ≈ 87 days
NUM_CANDLES = 75000  # Adjust based on how much history you want

# Trading parameters
INITIAL_BALANCE = 100
RISK_PER_TRADE = 5  # $5 risk per trade

# Strategy settings
REQUIRE_200_EMA = False  # Set to True to require price above/below 200 EMA
REQUIRE_RSI_FILTER = True  # Set to True to enable RSI filter
RSI_BULLISH_THRESHOLD = 60  # RSI must be >= this for bullish trades
RSI_BEARISH_THRESHOLD = 40  # RSI must be <= this for bearish trades
SL_BUFFER_PIPS = 2  # Buffer in pips for SL (prevents premature stops)

# ================= MT5 CONNECT =================
print("Connecting to MT5...")
if not mt5.initialize(
    login=int(os.getenv("MT5_LOGIN")),
    password=os.getenv("MT5_PASSWORD"),
    server=os.getenv("MT5_SERVER"),
):
    print("MT5 initialization failed:", mt5.last_error())
    exit()

print("✓ Connected to MT5")

# ================= INDICATORS =================
def ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_pip_value(symbol, price):
    """Calculate pip value for position sizing"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 10  # Default for EURUSD
    
    point = symbol_info.point
    
    if "JPY" in symbol:
        pip_size = 0.01
        pip_multiplier = 1
    else:
        pip_size = 0.0001
        pip_multiplier = 10
    
    # For USD pairs
    if symbol.endswith("USD"):
        pip_value_per_lot = pip_size * symbol_info.trade_contract_size
    elif symbol.startswith("USD"):
        pip_value_per_lot = (pip_size / price) * symbol_info.trade_contract_size
    else:
        pip_value_per_lot = (pip_size / price) * symbol_info.trade_contract_size
    
    return pip_value_per_lot

# ================= DATA LOADING =================
print(f"\nLoading data for {SYMBOL}...")

symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    raise RuntimeError(f"Symbol {SYMBOL} not found")

if not symbol_info.visible:
    mt5.symbol_select(SYMBOL, True)

# Get M5 historical data
print(f"Fetching {NUM_CANDLES} M5 candles...")
rates_m5 = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

if rates_m5 is None or len(rates_m5) == 0:
    print("❌ No M5 data returned")
    print("MT5 Error:", mt5.last_error())
    print("\nTroubleshooting:")
    print("1. Make sure MT5 terminal is running")
    print("2. Check if EURUSD is in Market Watch")
    print("3. Try reducing NUM_CANDLES (some brokers limit history)")
    mt5.shutdown()
    exit()

df = pd.DataFrame(rates_m5)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"✓ M5 Data loaded: {len(df)} candles")
print(f"  From: {df['time'].min()}")
print(f"  To:   {df['time'].max()}")

# Get M30 historical data for RSI
if REQUIRE_RSI_FILTER:
    print(f"\nFetching M30 candles for RSI filter...")
    rates_m30 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, NUM_CANDLES // 6 + 100)
    
    if rates_m30 is None or len(rates_m30) == 0:
        print("⚠ Warning: No M30 data returned, RSI filter will be disabled")
        REQUIRE_RSI_FILTER = False
        df_m30 = None
    else:
        df_m30 = pd.DataFrame(rates_m30)
        df_m30['time'] = pd.to_datetime(df_m30['time'], unit='s')
        print(f"✓ M30 Data loaded: {len(df_m30)} candles")
else:
    df_m30 = None

# Calculate indicators
print("\nCalculating indicators...")
df['ema_21'] = ema(df['close'], 21)
df['ema_51'] = ema(df['close'], 51)
df['ema_200'] = ema(df['close'], 200)

# Calculate M5 RSI
if REQUIRE_RSI_FILTER:
    df['rsi'] = rsi(df['close'], 14)
    print("✓ M5 RSI calculated")
    
    # Calculate M30 RSI
    if df_m30 is not None:
        df_m30['rsi'] = rsi(df_m30['close'], 14)
        print("✓ M30 RSI calculated")

print("✓ All indicators calculated")

# ================= BACKTEST =================
print("\n" + "="*70)
print("RUNNING BACKTEST")
print("="*70)

def get_m30_rsi_at_time(timestamp, df_m30):
    """Get M30 RSI value for a given M5 timestamp"""
    if df_m30 is None:
        return None
    
    # Find the M30 candle that contains this M5 timestamp
    m30_candle = df_m30[df_m30['time'] <= timestamp].iloc[-1] if len(df_m30[df_m30['time'] <= timestamp]) > 0 else None
    
    if m30_candle is None:
        return None
    
    return m30_candle['rsi']

balance = INITIAL_BALANCE
equity_curve = []
trade_log = []

wins = 0
losses = 0
total_profit = 0
total_loss = 0

in_trade = False
trade_entry = None
trade_direction = None
trade_sl = None
trade_entry_price = None
trade_lot_size = None

# Get point and pip info
point = symbol_info.point
pip_multiplier = 10 if "JPY" not in SYMBOL else 1

# Start from bar 201 to ensure all EMAs are valid
for i in range(201, len(df)):
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    
    # ========== MANAGE EXISTING POSITION ==========
    if in_trade:
        # Update trailing stop to current 51 EMA
        current_51_ema = curr['ema_51']
        
        # Add buffer
        buffer = SL_BUFFER_PIPS * point * pip_multiplier
        
        if trade_direction == "BUY":
            # For BUY, SL should be below 51 EMA
            new_sl = current_51_ema - buffer
            
            # Only update if it's higher than current SL (trailing up)
            if new_sl > trade_sl:
                trade_sl = new_sl
        
        else:  # SELL
            # For SELL, SL should be above 51 EMA
            new_sl = current_51_ema + buffer
            
            # Only update if it's lower than current SL (trailing down)
            if new_sl < trade_sl:
                trade_sl = new_sl
        
        # Check if SL was hit
        hit_sl = False
        
        if trade_direction == "BUY":
            if curr['low'] <= trade_sl:
                hit_sl = True
                exit_price = trade_sl
        else:  # SELL
            if curr['high'] >= trade_sl:
                hit_sl = True
                exit_price = trade_sl
        
        if hit_sl:
            # Calculate P&L
            if trade_direction == "BUY":
                pnl_pips = (exit_price - trade_entry_price) / (point * pip_multiplier)
            else:
                pnl_pips = (trade_entry_price - exit_price) / (point * pip_multiplier)
            
            pnl_dollars = pnl_pips * trade_lot_size * calculate_pip_value(SYMBOL, trade_entry_price)
            
            balance += pnl_dollars
            
            if pnl_dollars > 0:
                wins += 1
                total_profit += pnl_dollars
            else:
                losses += 1
                total_loss += abs(pnl_dollars)
            
            # Log trade
            trade_log.append({
                'entry_time': trade_entry,
                'exit_time': curr['time'],
                'direction': trade_direction,
                'entry_price': trade_entry_price,
                'exit_price': exit_price,
                'pnl_pips': pnl_pips,
                'pnl_dollars': pnl_dollars,
                'balance': balance
            })
            
            in_trade = False
            trade_entry = None
            trade_direction = None
            trade_sl = None
            trade_entry_price = None
            trade_lot_size = None
        
        equity_curve.append(balance)
        continue
    
    # ========== LOOK FOR NEW SIGNALS ==========
    
    # Bullish crossover
    bullish = (
        prev['ema_21'] <= prev['ema_51'] and
        curr['ema_21'] > curr['ema_51']
    )
    
    # Bearish crossover
    bearish = (
        prev['ema_21'] >= prev['ema_51'] and
        curr['ema_21'] < curr['ema_51']
    )
    
    # Apply 200 EMA filter if enabled
    if REQUIRE_200_EMA:
        if bullish:
            bullish = curr['ema_21'] > curr['ema_200'] and curr['ema_51'] > curr['ema_200']
        if bearish:
            bearish = curr['ema_21'] < curr['ema_200'] and curr['ema_51'] < curr['ema_200']
    
    # Apply RSI filter if enabled
    if REQUIRE_RSI_FILTER and (bullish or bearish):
        m5_rsi = curr['rsi']
        m30_rsi = get_m30_rsi_at_time(curr['time'], df_m30)
        
        if pd.isna(m5_rsi) or (df_m30 is not None and (m30_rsi is None or pd.isna(m30_rsi))):
            # Skip if RSI not available
            bullish = False
            bearish = False
        else:
            if bullish:
                # For bullish trades: both M5 and M30 RSI must be >= threshold
                if df_m30 is not None:
                    bullish = m5_rsi >= RSI_BULLISH_THRESHOLD and m30_rsi >= RSI_BULLISH_THRESHOLD
                else:
                    bullish = m5_rsi >= RSI_BULLISH_THRESHOLD
            
            if bearish:
                # For bearish trades: both M5 and M30 RSI must be <= threshold
                if df_m30 is not None:
                    bearish = m5_rsi <= RSI_BEARISH_THRESHOLD and m30_rsi <= RSI_BEARISH_THRESHOLD
                else:
                    bearish = m5_rsi <= RSI_BEARISH_THRESHOLD
    
    if not bullish and not bearish:
        equity_curve.append(balance)
        continue
    
    # ========== ENTER TRADE ==========
    entry_price = curr['close']
    
    # Calculate initial SL at 51 EMA with buffer
    buffer = SL_BUFFER_PIPS * point * pip_multiplier
    
    if bullish:
        initial_sl = curr['ema_51'] - buffer
        direction = "BUY"
    else:
        initial_sl = curr['ema_51'] + buffer
        direction = "SELL"
    
    # Validate SL
    if (bullish and initial_sl >= entry_price) or (bearish and initial_sl <= entry_price):
        equity_curve.append(balance)
        continue
    
    # Calculate position size based on risk
    sl_dist_pips = abs(entry_price - initial_sl) / (point * pip_multiplier)
    pip_value = calculate_pip_value(SYMBOL, entry_price)
    
    if sl_dist_pips > 0 and pip_value > 0:
        lot_size = RISK_PER_TRADE / (sl_dist_pips * pip_value)
        lot_size = round(lot_size, 2)  # Round to 2 decimals
        
        # Apply min/max lot size
        if lot_size < symbol_info.volume_min:
            lot_size = symbol_info.volume_min
        elif lot_size > symbol_info.volume_max:
            lot_size = symbol_info.volume_max
    else:
        equity_curve.append(balance)
        continue
    
    # Enter trade
    in_trade = True
    trade_entry = curr['time']
    trade_direction = direction
    trade_sl = initial_sl
    trade_entry_price = entry_price
    trade_lot_size = lot_size
    
    equity_curve.append(balance)

# ================= RESULTS =================
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
avg_win = (total_profit / wins) if wins > 0 else 0
avg_loss = (total_loss / losses) if losses > 0 else 0
profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
net_profit = balance - INITIAL_BALANCE
return_pct = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)
print(f"Symbol              : {SYMBOL}")
print(f"Timeframe           : M5")
print(f"Period              : {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
print(f"Total Candles       : {len(df)}")
print(f"")
print(f"Filters Enabled:")
print(f"  200 EMA Filter    : {'Yes' if REQUIRE_200_EMA else 'No'}")
print(f"  RSI Filter        : {'Yes' if REQUIRE_RSI_FILTER else 'No'}")
if REQUIRE_RSI_FILTER:
    print(f"  RSI Bullish (>=)  : {RSI_BULLISH_THRESHOLD}")
    print(f"  RSI Bearish (<=)  : {RSI_BEARISH_THRESHOLD}")
    print(f"  M30 RSI Required  : Yes")
print("-"*70)
print(f"Initial Balance     : ${INITIAL_BALANCE:.2f}")
print(f"Final Balance       : ${balance:.2f}")
print(f"Net Profit/Loss     : ${net_profit:.2f}")
print(f"Return              : {return_pct:.2f}%")
print("-"*70)
print(f"Total Trades        : {total_trades}")
print(f"Winning Trades      : {wins}")
print(f"Losing Trades       : {losses}")
print(f"Win Rate            : {win_rate:.2f}%")
print("-"*70)
print(f"Total Profit        : ${total_profit:.2f}")
print(f"Total Loss          : ${total_loss:.2f}")
print(f"Average Win         : ${avg_win:.2f}")
print(f"Average Loss        : ${avg_loss:.2f}")
print(f"Profit Factor       : {profit_factor:.2f}")
print(f"Risk per Trade      : ${RISK_PER_TRADE}")
print("="*70)

# ================= TRADE LOG =================
if trade_log:
    print("\n" + "="*70)
    print("RECENT TRADES (Last 10)")
    print("="*70)
    
    trades_df = pd.DataFrame(trade_log)
    
    # Show last 10 trades
    for idx, trade in trades_df.tail(10).iterrows():
        print(f"\nTrade #{idx + 1}")
        print(f"  Direction   : {trade['direction']}")
        print(f"  Entry Time  : {trade['entry_time']}")
        print(f"  Exit Time   : {trade['exit_time']}")
        print(f"  Entry Price : {trade['entry_price']:.5f}")
        print(f"  Exit Price  : {trade['exit_price']:.5f}")
        print(f"  P&L (pips)  : {trade['pnl_pips']:.1f}")
        print(f"  P&L ($)     : ${trade['pnl_dollars']:.2f}")
        print(f"  Balance     : ${trade['balance']:.2f}")
    
    # Save full trade log to CSV
    trades_df.to_csv(f"backtest_trades_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print(f"\n✓ Full trade log saved to CSV")

# ================= EQUITY CURVE PLOT =================
if equity_curve:
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, linewidth=2)
    plt.title(f'Equity Curve - {SYMBOL} (51 EMA Trailing Stop)', fontsize=14, fontweight='bold')
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Balance ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=INITIAL_BALANCE, color='r', linestyle='--', label='Initial Balance')
    plt.legend()
    
    filename = f"equity_curve_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Equity curve saved as {filename}")
    
    plt.show()

mt5.shutdown()
print("\n✓ Backtest complete")