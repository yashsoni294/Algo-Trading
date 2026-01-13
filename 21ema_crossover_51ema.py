"""
This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range. And the stop loss is 1.5 ATR 
with position sizing based on risk amount ($5 risk, $10 profit per trade).
"""


import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import time
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def connect_mt5(portable=True):
    """Connect to MT5 with portable mode option"""
    
    if portable:
        mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
        # Alternative paths - uncomment if needed:
        # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
        # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
        if not os.path.exists(mt5_path):
            #print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
            portable = False
    
    if portable:
        if not mt5.initialize(
            path=mt5_path,
            login=5044597561,
            password="Ej@6UjSs",
            server="MetaQuotes-Demo",
            timeout=60000,
            portable=True
        ):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}")
    else:
        if not mt5.initialize(
            login=5044597561,
            password="Ej@6UjSs",
            server="MetaQuotes-Demo"
        ):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}")
    
    account_info = mt5.account_info()
    if account_info:
        #print("✓ Connected to MT5")
        #print(f"  Account: {account_info.login}")
        #print(f"  Balance: ${account_info.balance}")
        #print(f"  Leverage: 1:{account_info.leverage}")
        pass
    else:
        #print("✓ Connected to MT5")
        pass


def shutdown_mt5():
    mt5.shutdown()
    #print("MT5 disconnected")


def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=100):
    """Get candle data for analysis"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_ema(df, period):
    """Calculate Exponential Moving Average"""
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_ema_crossover(df, rsi_threshold_bullish=60, rsi_threshold_bearish=40):
    """
    Detect EMA crossover signals WITH RSI filter
    - Bullish: 21 EMA crosses above 51 EMA AND RSI >= 60
    - Bearish: 21 EMA crosses below 51 EMA AND RSI <= 40
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if len(df) < 52:  # Need at least 52 candles for 51 EMA
        return "neutral", None, None
    
    # Calculate EMAs
    df['ema_21'] = calculate_ema(df, 21)
    df['ema_51'] = calculate_ema(df, 51)
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df, period=14)
    
    # Get last 3 candles for crossover detection
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Check for bullish crossover (21 EMA crosses above 51 EMA) + RSI >= 60
    if (prev['ema_21'] <= prev['ema_51'] and 
        curr['ema_21'] > curr['ema_51'] and
        curr['rsi'] >= rsi_threshold_bullish):
        return "bullish", curr, df
    
    # Check for bearish crossover (21 EMA crosses below 51 EMA) + RSI <= 40
    if (prev['ema_21'] >= prev['ema_51'] and 
        curr['ema_21'] < curr['ema_51'] and
        curr['rsi'] <= rsi_threshold_bearish):
        return "bearish", curr, df
    
    # Check if crossover occurred but RSI filter not met
    if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
        #print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
        pass
    
    if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
        #print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
        pass
    
    return "neutral", curr, df

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def place_ema_trade(symbol, signal, curr_candle, df, risk_amount=5.0, reward_amount=10.0):
    """
    Place trade based on EMA crossover + RSI filter
    SL: Current candle's high/low +/- (1.5 × ATR)
    TP: Based on fixed reward amount
    Lot size: Calculated based on risk amount
    
    Args:
        risk_amount: Dollar amount willing to risk per trade (default: $5)
        reward_amount: Dollar amount target profit per trade (default: $10)
    """
    if signal not in ["bullish", "bearish"]:
        #print("No trade signal (neutral) — skipping trade.")
        return

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        #print(f"Failed to get symbol info for {symbol}")
        return
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            #print(f"Failed to add {symbol} to Market Watch")
            return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        #print(f"Failed to get tick for {symbol}")
        return
    
    # Entry price
    price = tick.ask if signal == "bullish" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

    # Calculate ATR (14 period)
    df['atr'] = calculate_atr(df, period=14)
    current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
    if pd.isna(current_atr) or current_atr <= 0:
        #print(f"⚠ Invalid ATR value: {current_atr}, cannot place trade")
        return
    
    # ATR buffer (1.5 × ATR)
    atr_buffer = 1.5 * current_atr
    
    # Calculate SL based on current candle + ATR buffer
    if signal == "bullish":
        sl_price = curr_candle['low'] - atr_buffer
        sl_dist_price = abs(price - sl_price)
    else:
        sl_price = curr_candle['high'] + atr_buffer
        sl_dist_price = abs(price - sl_price)
    
    # Round SL to proper digits
    sl = round(sl_price, symbol_info.digits)
    
    # Validate SL
    if signal == "bullish":
        if sl >= price:
            #print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
            return
    else:
        if sl <= price:
            #print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
            return
    
    # Get contract size and point value
    contract_size = symbol_info.trade_contract_size
    point = symbol_info.point
    
    # Calculate pip value per lot
    # For forex pairs, 1 pip = 10 points (except JPY pairs where 1 pip = 100 points)
    if "JPY" in symbol:
        pip_multiplier = 100
    else:
        pip_multiplier = 10
    
    # SL distance in pips
    sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
    # Calculate pip value for 1 standard lot (100,000 units)
    # For USD-based pairs or when account currency is USD
    if symbol.startswith("USD"):
        # USD is base currency (e.g., USDJPY, USDCHF)
        # Pip value = (pip in quote currency) * contract_size
        if "JPY" in symbol:
            pip_value_per_lot = (0.01) * contract_size / price  # For JPY pairs
        else:
            pip_value_per_lot = (0.0001) * contract_size / price
    elif symbol.endswith("USD"):
        # USD is quote currency (e.g., EURUSD, GBPUSD)
        if "JPY" in symbol:
            pip_value_per_lot = 0.01 * contract_size
        else:
            pip_value_per_lot = 0.0001 * contract_size
    else:
        # Cross pairs - simplified calculation
        pip_value_per_lot = 0.0001 * contract_size / price
    
    # Calculate required lot size for desired risk
    # risk_amount = lot_size × sl_dist_pips × pip_value_per_lot
    if sl_dist_pips > 0 and pip_value_per_lot > 0:
        lot_size = risk_amount / (sl_dist_pips * pip_value_per_lot)
    else:
        #print(f"⚠ Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
        return
    
    # Round to broker's volume step (usually 0.01)
    volume_step = symbol_info.volume_step
    lot_size = round(lot_size / volume_step) * volume_step
    
    # Apply min/max lot size limits
    if lot_size < symbol_info.volume_min:
        #print(f"⚠ Calculated lot size {lot_size:.2f} is below minimum {symbol_info.volume_min}")
        lot_size = symbol_info.volume_min
    elif lot_size > symbol_info.volume_max:
        #print(f"⚠ Calculated lot size {lot_size:.2f} exceeds maximum {symbol_info.volume_max}")
        lot_size = symbol_info.volume_max
    
    # Calculate TP based on reward amount
    # reward_amount = lot_size × tp_dist_pips × pip_value_per_lot
    tp_dist_pips = reward_amount / (lot_size * pip_value_per_lot)
    tp_dist_price = tp_dist_pips * (point * pip_multiplier)
    
    # Calculate final TP
    if signal == "bullish":
        tp = round(price + tp_dist_price, symbol_info.digits)
    else:
        tp = round(price - tp_dist_price, symbol_info.digits)
    
    # Validate TP
    if signal == "bullish":
        if tp <= price:
            #print("ERROR: Invalid take profit for bullish trade")
            return
    else:
        if tp >= price:
            #print("ERROR: Invalid take profit for bearish trade")
            return
    
    # Calculate actual risk/reward in dollars
    actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
    actual_reward = lot_size * tp_dist_pips * pip_value_per_lot
    
    #print(f"\n{'='*70}")
    #print(f"TRADE SETUP - {symbol}")
    #print(f"{'='*70}")
    #print(f"Signal: {signal.upper()}")
    #print(f"Entry Price: {price:.5f}")
    #print(f"Stop Loss: {sl:.5f} (Distance: {sl_dist_pips:.1f} pips)")
    #print(f"Take Profit: {tp:.5f} (Distance: {tp_dist_pips:.1f} pips)")
    #print(f"")
    #print(f"Position Size: {lot_size:.2f} lots")
    #print(f"Contract Size: {contract_size:,.0f}")
    #print(f"Pip Value/Lot: ${pip_value_per_lot:.2f}")
    #print(f"")
    #print(f"Expected RISK: ${actual_risk:.2f} (Target: ${risk_amount:.2f})")
    #print(f"Expected REWARD: ${actual_reward:.2f} (Target: ${reward_amount:.2f})")
    #print(f"Risk/Reward Ratio: 1:{actual_reward/actual_risk:.2f}")
    #print(f"")
    #print(f"ATR(14): {current_atr:.5f} | ATR Buffer (1.5×): {atr_buffer:.5f}")
    #print(f"{'='*70}\n")

    # Try different filling modes
    filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
    for filling_type in filling_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 234001,
            "comment": f"EMA+RSI {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        
        # Check if result is None
        if result is None:
            error = mt5.last_error()
            #print(f"✗ Order send failed (returned None) with {filling_type}")
            #print(f"  MT5 Error: {error}")
            continue
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            #print(f"✓ {signal.upper()} trade placed successfully!")
            #print(f"  Order Ticket: {result.order}")
            #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
            #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
            #print(f"  RSI: {curr_candle['rsi']:.2f}")
            return
        elif result.retcode == 10018:
            #print(f"✗ Market is closed for {symbol}")
            return
        elif result.retcode != 10030:
            #print(f"✗ Order failed: {result.retcode} - {result.comment}")
            return
    
    #print(f"✗ Order failed with all filling modes.")


def is_market_open(symbol: str) -> bool:
    """Check if market is open and tradable"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False

    if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    if tick.bid <= 0 or tick.ask <= 0:
        return False

    # Check tick freshness
    tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now = datetime.now(timezone.utc)

    if (now - tick_time).total_seconds() > 60:
        return False

    # Avoid rollover period
    if (tick_time.hour == 23 and tick_time.minute >= 55) or \
       (tick_time.hour == 0 and tick_time.minute <= 5):
        return False

    return True


def check_existing_position(symbol):
    """Check if there's already an open position for this symbol"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return len(positions) > 0


def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5, risk_per_trade=5.0, reward_per_trade=10.0):
    """Run the EMA crossover strategy with RSI filter and risk-based position sizing"""
    
    # Check for existing position
    if check_existing_position(symbol):
        #print(f"[{symbol}] Already have open position - skipping")
        return
    
    # Get data
    df = get_candle_data(symbol, timeframe, n=100)
    if df is None or len(df) < 52:
        #print(f"[{symbol}] Insufficient data")
        return
    
    # Detect signal (with RSI filter)
    signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
    if signal == "neutral":
        #print(f"[{symbol}] No valid signal")
        return
    
    #print(f"\n[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
    #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
    #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
    #print(f"  RSI: {curr_candle['rsi']:.2f}")
    #print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
    # f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
    # Place trade with risk-based position sizing
    place_ema_trade(symbol, signal, curr_candle, df_with_emas, 
                   risk_amount=risk_per_trade, reward_amount=reward_per_trade)


def scan_markets():
    """
    Scan all currency pairs for trading signals.
    This function is called by the scheduler.
    """
    current_time = datetime.now()
    #print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
    
    for pair in currency_pair_list:
        try:
            if not is_market_open(pair):
                #print(f"[{pair}] Market not open - skipping")
                continue
            
            # Get current candle time
            df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
            if df is not None:
                current_candle_time = df.iloc[-1]['time']
                
                # Only check if new candle formed
                if pair not in last_check or last_check[pair] != current_candle_time:
                    run_ema_strategy(pair, mt5.TIMEFRAME_M5, 
                                   risk_per_trade=RISK_PER_TRADE, 
                                   reward_per_trade=REWARD_PER_TRADE)
                    last_check[pair] = current_candle_time
            
        except Exception as e:
            #print(f"[{pair}] Error: {e}")
            import traceback
            # traceback.#print_exc()
            continue
    
    #print("\n" + "-" * 100)


# ===================== MAIN EXECUTION =====================

# Configuration
ACCOUNT_BALANCE = 100  # dollars
RISK_PER_TRADE = 5     # dollars ($5 risk per trade = 5% of $100)
REWARD_PER_TRADE = 10  # dollars (1:2 risk/reward)

currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
last_check = {}  # Track last candle time to avoid duplicate signals (global for scheduler)

#print("=" * 70)
#print("EMA CROSSOVER + RSI FILTER STRATEGY (RISK-BASED POSITION SIZING)")
#print("=" * 70)
#print(f"Account Balance: ${ACCOUNT_BALANCE}")
#print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
#print(f"Reward per Trade: ${REWARD_PER_TRADE} (1:{REWARD_PER_TRADE/RISK_PER_TRADE:.0f} R/R)")
#print(f"Strategy: 21/51 EMA on 5-Min Chart + RSI(14)")
#print(f"Entry: Bullish (RSI >= 60) | Bearish (RSI <= 40)")
#print(f"Stop Loss: Candle High/Low +/- 1.5× ATR(14)")
#print("=" * 70)

connect_mt5(portable=True)  # Use portable=False if path issues

# Create scheduler
scheduler = BlockingScheduler()

# Schedule the scan_markets function to run every minute
# This will check for new 5-minute candles efficiently
scheduler.add_job(
    scan_markets,
    trigger='cron',
    minute='*',  # Run every minute
    second='5',  # Start 5 seconds into each minute
    id='market_scanner',
    name='Scan markets for EMA+RSI signals',
    max_instances=1  # Prevent overlapping executions
)

#print("\n✓ Scheduler initialized")
#print("  Job: Scan markets every minute")
#print("  Press Ctrl+C to stop\n")

try:
    # Start the scheduler (blocking)
    scheduler.start()

except (KeyboardInterrupt, SystemExit):
    #print("\n\nStopping strategy...")
    scheduler.shutdown()

finally:
    shutdown_mt5()


"""
This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range. And the stop loss is 1.5 ATR 
with position sizing based on risk amount ($5 risk, $10 profit per trade).
"""


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os


# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             #print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo",
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo"
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         #print("Connected to MT5")
#         #print(f"Account: {account_info.login}")
#         #print(f"Balance: ${account_info.balance}")
#         #print(f"Leverage: 1:{account_info.leverage}")
#     else:
#         #print("Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     #print("MT5 disconnected")


# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=100):
#     """Get candle data for analysis"""
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
#     if rates is None or len(rates) == 0:
#         return None
    
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df


# def calculate_ema(df, period):
#     """Calculate Exponential Moving Average"""
#     return df['close'].ewm(span=period, adjust=False).mean()


# def calculate_rsi(df, period=14):
#     """Calculate Relative Strength Index (RSI)"""
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi


# def detect_ema_crossover(df, rsi_threshold_bullish=60, rsi_threshold_bearish=40):
#     """
#     Detect EMA crossover signals WITH RSI filter
#     - Bullish: 21 EMA crosses above 51 EMA AND RSI >= 60
#     - Bearish: 21 EMA crosses below 51 EMA AND RSI <= 40
#     Returns: 'bullish', 'bearish', or 'neutral'
#     """
#     if len(df) < 52:  # Need at least 52 candles for 51 EMA
#         return "neutral", None, None
    
#     # Calculate EMAs
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
    
#     # Calculate RSI
#     df['rsi'] = calculate_rsi(df, period=14)
    
#     # Get last 3 candles for crossover detection
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
#     prev2 = df.iloc[-3]
    
#     # Check for bullish crossover (21 EMA crosses above 51 EMA) + RSI >= 60
#     if (prev['ema_21'] <= prev['ema_51'] and 
#         curr['ema_21'] > curr['ema_51'] and
#         curr['rsi'] >= rsi_threshold_bullish):
#         return "bullish", curr, df
    
#     # Check for bearish crossover (21 EMA crosses below 51 EMA) + RSI <= 40
#     if (prev['ema_21'] >= prev['ema_51'] and 
#         curr['ema_21'] < curr['ema_51'] and
#         curr['rsi'] <= rsi_threshold_bearish):
#         return "bearish", curr, df
    
#     # Check if crossover occurred but RSI filter not met
#     if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
#         #print(f"Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         #print(f"Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
#     return "neutral", curr, df


# def calculate_atr(df, period=14):
#     """Calculate Average True Range (ATR)"""
#     high = df['high']
#     low = df['low']
#     close = df['close']
    
#     # True Range calculation
#     tr1 = high - low
#     tr2 = abs(high - close.shift())
#     tr3 = abs(low - close.shift())
    
#     tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#     atr = tr.rolling(window=period).mean()
    
#     return atr


# def place_ema_trade(symbol, signal, curr_candle, df, risk_amount=5.0, reward_amount=10.0):
#     """
#     Place trade based on EMA crossover + RSI filter
#     SL: Current candle's high/low +/- (1.5 × ATR)
#     TP: Based on fixed reward amount
#     Lot size: Calculated based on risk amount
    
#     Args:
#         risk_amount: Dollar amount willing to risk per trade (default: $5)
#         reward_amount: Dollar amount target profit per trade (default: $10)
#     """
#     if signal not in ["bullish", "bearish"]:
#         #print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         #print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             #print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         #print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         #print(f"Invalid ATR value: {current_atr}, cannot place trade")
#         return
    
#     # ATR buffer (1.5 × ATR)
#     atr_buffer = 1.5 * current_atr
    
#     # Calculate SL based on current candle + ATR buffer
#     if signal == "bullish":
#         sl_price = curr_candle['low'] - atr_buffer
#         sl_dist_price = abs(price - sl_price)
#     else:
#         sl_price = curr_candle['high'] + atr_buffer
#         sl_dist_price = abs(price - sl_price)
    
#     # Round SL to proper digits
#     sl = round(sl_price, symbol_info.digits)
    
#     # Validate SL
#     if signal == "bullish":
#         if sl >= price:
#             #print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
#             return
#     else:
#         if sl <= price:
#             #print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
#             return
    
#     # Get contract size and point value
#     contract_size = symbol_info.trade_contract_size
#     point = symbol_info.point
    
#     # Calculate pip value per lot
#     # For forex pairs, 1 pip = 10 points (except JPY pairs where 1 pip = 100 points)
#     if "JPY" in symbol:
#         pip_multiplier = 100
#     else:
#         pip_multiplier = 10
    
#     # SL distance in pips
#     sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
#     # Calculate pip value for 1 standard lot (100,000 units)
#     # For USD-based pairs or when account currency is USD
#     if symbol.startswith("USD"):
#         # USD is base currency (e.g., USDJPY, USDCHF)
#         # Pip value = (pip in quote currency) * contract_size
#         if "JPY" in symbol:
#             pip_value_per_lot = (0.01) * contract_size / price  # For JPY pairs
#         else:
#             pip_value_per_lot = (0.0001) * contract_size / price
#     elif symbol.endswith("USD"):
#         # USD is quote currency (e.g., EURUSD, GBPUSD)
#         if "JPY" in symbol:
#             pip_value_per_lot = 0.01 * contract_size
#         else:
#             pip_value_per_lot = 0.0001 * contract_size
#     else:
#         # Cross pairs - simplified calculation
#         pip_value_per_lot = 0.0001 * contract_size / price
    
#     # Calculate required lot size for desired risk
#     # risk_amount = lot_size × sl_dist_pips × pip_value_per_lot
#     if sl_dist_pips > 0 and pip_value_per_lot > 0:
#         lot_size = risk_amount / (sl_dist_pips * pip_value_per_lot)
#     else:
#         #print(f"Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
#         return
    
#     # Round to broker's volume step (usually 0.01)
#     volume_step = symbol_info.volume_step
#     lot_size = round(lot_size / volume_step) * volume_step
    
#     # Apply min/max lot size limits
#     if lot_size < symbol_info.volume_min:
#         #print(f"Calculated lot size {lot_size:.2f} is below minimum {symbol_info.volume_min}")
#         lot_size = symbol_info.volume_min
#     elif lot_size > symbol_info.volume_max:
#         #print(f"Calculated lot size {lot_size:.2f} exceeds maximum {symbol_info.volume_max}")
#         lot_size = symbol_info.volume_max
    
#     # Calculate TP based on reward amount
#     # reward_amount = lot_size × tp_dist_pips × pip_value_per_lot
#     tp_dist_pips = reward_amount / (lot_size * pip_value_per_lot)
#     tp_dist_price = tp_dist_pips * (point * pip_multiplier)
    
#     # Calculate final TP
#     if signal == "bullish":
#         tp = round(price + tp_dist_price, symbol_info.digits)
#     else:
#         tp = round(price - tp_dist_price, symbol_info.digits)
    
#     # Validate TP
#     if signal == "bullish":
#         if tp <= price:
#             #print("ERROR: Invalid take profit for bullish trade")
#             return
#     else:
#         if tp >= price:
#             #print("ERROR: Invalid take profit for bearish trade")
#             return
    
#     # Calculate actual risk/reward in dollars
#     actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
#     actual_reward = lot_size * tp_dist_pips * pip_value_per_lot
    
#     #print(f"\n{'='*70}")
#     #print(f"TRADE SETUP - {symbol}")
#     #print(f"{'='*70}")
#     #print(f"Signal: {signal.upper()}")
#     #print(f"Entry Price: {price:.5f}")
#     #print(f"Stop Loss: {sl:.5f} (Distance: {sl_dist_pips:.1f} pips)")
#     #print(f"Take Profit: {tp:.5f} (Distance: {tp_dist_pips:.1f} pips)")
#     #print(f"Position Size: {lot_size:.2f} lots")
#     #print(f"Contract Size: {contract_size:,.0f}")
#     #print(f"Pip Value/Lot: ${pip_value_per_lot:.2f}")
#     #print(f"Expected RISK: ${actual_risk:.2f} (Target: ${risk_amount:.2f})")
#     #print(f"Expected REWARD: ${actual_reward:.2f} (Target: ${reward_amount:.2f})")
#     #print(f"Risk/Reward Ratio: 1:{actual_reward/actual_risk:.2f}")
#     #print(f"ATR(14): {current_atr:.5f} | ATR Buffer (1.5×): {atr_buffer:.5f}")
#     #print(f"{'='*70}\n")

#     # Try different filling modes
#     filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
#     for filling_type in filling_modes:
#         request = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": symbol,
#             "volume": lot_size,
#             "type": order_type,
#             "price": price,
#             "sl": float(sl),
#             "tp": float(tp),
#             "deviation": 20,
#             "magic": 234001,
#             "comment": f"EMA+RSI {signal}",
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": filling_type,
#         }

#         result = mt5.order_send(request)
        
#         # Check if result is None
#         if result is None:
#             error = mt5.last_error()
#             #print(f"Order send failed (returned None) with {filling_type}")
#             #print(f"MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             #print(f"{signal.upper()} trade placed successfully!")
#             #print(f"  Order Ticket: {result.order}")
#             #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             #print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             #print(f"Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             #print(f"Order failed: {result.retcode} - {result.comment}")
#             return
    
#     #print("Order failed with all filling modes.")


# def is_market_open(symbol: str) -> bool:
#     """Check if market is open and tradable"""
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         return False

#     if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
#         return False

#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             return False

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         return False

#     if tick.bid <= 0 or tick.ask <= 0:
#         return False

#     # Check tick freshness
#     tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
#     now = datetime.now(timezone.utc)

#     if (now - tick_time).total_seconds() > 60:
#         return False

#     # Avoid rollover period
#     if (tick_time.hour == 23 and tick_time.minute >= 55) or \
#        (tick_time.hour == 0 and tick_time.minute <= 5):
#         return False

#     return True


# def check_existing_position(symbol):
#     """Check if there's already an open position for this symbol"""
#     positions = mt5.positions_get(symbol=symbol)
#     if positions is None:
#         return False
#     return len(positions) > 0


# def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5, risk_per_trade=5.0, reward_per_trade=10.0):
#     """Run the EMA crossover strategy with RSI filter and risk-based position sizing"""
    
#     # Check for existing position
#     if check_existing_position(symbol):
#         #print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         #print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         #print(f"[{symbol}] No valid signal")
#         return
    
#     #print(f"\n[{symbol}]  {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     #print(f"  RSI: {curr_candle['rsi']:.2f}")
#     #print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade with risk-based position sizing
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas, 
#                    risk_amount=risk_per_trade, reward_amount=reward_per_trade)


# # ===================== MAIN EXECUTION =====================

# # Configuration
# ACCOUNT_BALANCE = 100  # dollars
# RISK_PER_TRADE = 5     # dollars ($5 risk per trade = 5% of $100)
# REWARD_PER_TRADE = 10  # dollars (1:2 risk/reward)

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# #print("=" * 123)
# #print("EMA CROSSOVER + RSI FILTER STRATEGY (RISK-BASED POSITION SIZING)")
# #print("=" * 123)
# #print(f"Account Balance: ${ACCOUNT_BALANCE}")
# #print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
# #print(f"Reward per Trade: ${REWARD_PER_TRADE} (1:{REWARD_PER_TRADE/RISK_PER_TRADE:.0f} R/R)")
# #print("Strategy: 21/51 EMA on 5-Min Chart + RSI(14)")
# #print("Entry: Bullish (RSI >= 60) | Bearish (RSI <= 40)")
# #print("Stop Loss: Candle High/Low +/- 1.5× ATR(14)")
# #print("=" * 123)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         #print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     #print(f"[{pair}] Market not open - skipping")
#                     continue
                
#                 # Get current candle time
#                 df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#                 if df is not None:
#                     current_candle_time = df.iloc[-1]['time']
                    
#                     # Only check if new candle formed
#                     if pair not in last_check or last_check[pair] != current_candle_time:
#                         run_ema_strategy(pair, mt5.TIMEFRAME_M5, 
#                                        risk_per_trade=RISK_PER_TRADE, 
#                                        reward_per_trade=REWARD_PER_TRADE)
#                         last_check[pair] = current_candle_time
                
#             except Exception as e:
#                 #print(f"[{pair}] Error: {e}")
#                 import traceback
#                 traceback.#print_exc()
#                 continue
        
#         # Wait for next check (check every 60 seconds)
#         #print("\n" + "-" * 123)
#         time.sleep(60)

# except KeyboardInterrupt:
#     #print("\n\nStopping strategy...")

# finally:
#     shutdown_mt5()


"""
This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range. And the stop loss is 1.5 ATR 
with 1 lot size as default.
"""


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os


# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             #print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo",
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo"
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         #print("✓ Connected to MT5")
#         #print(f"  Account: {account_info.login}")
#         #print(f"  Balance: ${account_info.balance}")
#     else:
#         #print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     #print("MT5 disconnected")


# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=100):
#     """Get candle data for analysis"""
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
#     if rates is None or len(rates) == 0:
#         return None
    
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df


# def calculate_ema(df, period):
#     """Calculate Exponential Moving Average"""
#     return df['close'].ewm(span=period, adjust=False).mean()


# def calculate_rsi(df, period=14):
#     """Calculate Relative Strength Index (RSI)"""
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi


# def detect_ema_crossover(df, rsi_threshold_bullish=60, rsi_threshold_bearish=40):
#     """
#     Detect EMA crossover signals WITH RSI filter
#     - Bullish: 21 EMA crosses above 51 EMA AND RSI >= 60
#     - Bearish: 21 EMA crosses below 51 EMA AND RSI <= 40
#     Returns: 'bullish', 'bearish', or 'neutral'
#     """
#     if len(df) < 52:  # Need at least 52 candles for 51 EMA
#         return "neutral", None, None
    
#     # Calculate EMAs
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
    
#     # Calculate RSI
#     df['rsi'] = calculate_rsi(df, period=14)
    
#     # Get last 3 candles for crossover detection
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
#     prev2 = df.iloc[-3]
    
#     # Check for bullish crossover (21 EMA crosses above 51 EMA) + RSI >= 60
#     if (prev['ema_21'] <= prev['ema_51'] and 
#         curr['ema_21'] > curr['ema_51'] and
#         curr['rsi'] >= rsi_threshold_bullish):
#         return "bullish", curr, df
    
#     # Check for bearish crossover (21 EMA crosses below 51 EMA) + RSI <= 40
#     if (prev['ema_21'] >= prev['ema_51'] and 
#         curr['ema_21'] < curr['ema_51'] and
#         curr['rsi'] <= rsi_threshold_bearish):
#         return "bearish", curr, df
    
#     # Check if crossover occurred but RSI filter not met
#     if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
#         #print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         #print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
#     return "neutral", curr, df

# def calculate_atr(df, period=14):
#     """Calculate Average True Range (ATR)"""
#     high = df['high']
#     low = df['low']
#     close = df['close']
    
#     # True Range calculation
#     tr1 = high - low
#     tr2 = abs(high - close.shift())
#     tr3 = abs(low - close.shift())
    
#     tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#     atr = tr.rolling(window=period).mean()
    
#     return atr


# def place_ema_trade(symbol, signal, curr_candle, df, lot=0.3):
#     """
#     Place trade based on EMA crossover + RSI filter
#     SL: Current candle's high/low +/- (1.5 × ATR)
#     TP: 2x SL distance
#     """
#     if signal not in ["bullish", "bearish"]:
#         #print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         #print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             #print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         #print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         #print(f"⚠ Invalid ATR value: {current_atr}, cannot place trade")
#         return
    
#     # ATR buffer (1.5 × ATR)
#     atr_buffer = 1.5 * current_atr
    
#     # Calculate SL based on current candle + ATR buffer
#     if signal == "bullish":
#         sl_price = curr_candle['low'] - atr_buffer
#         sl_dist = abs(price - sl_price)
#     else:
#         sl_price = curr_candle['high'] + atr_buffer
#         sl_dist = abs(price - sl_price)
    
#     # Set MINIMUM stop distances
#     point = symbol_info.point
    
#     # if symbol in ['USDCNH', 'USDZAR', 'USDTRY']:
#     #     min_pips = 30
#     # else:
#     #     min_pips = 15
    
#     # min_distance = min_pips * point * 10
    
#     #print(f"Symbol: {symbol}, Point: {point}")
#     #print(f"ATR(14): {current_atr:.5f}")
#     #print(f"ATR Buffer (1.5×): {atr_buffer:.5f}")
#     #print(f"Candle {'Low' if signal == 'bullish' else 'High'}: {curr_candle['low'] if signal == 'bullish' else curr_candle['high']:.5f}")
#     #print(f"Calculated SL: {sl_price:.5f}")
#     #print(f"Calculated SL dist: {sl_dist:.5f}")
#     # #print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     # if sl_dist < min_distance:
#     #     #print("⚠ SL too tight, using minimum distance")
#     #     sl_dist = min_distance
#     #     # Recalculate SL with minimum distance
#     #     if signal == "bullish":
#     #         sl_price = price - sl_dist
#     #     else:
#     #         sl_price = price + sl_dist
    
#     # TP is 2x SL distance
#     tp_dist = 2 * sl_dist
    
#     # Calculate final SL/TP
#     if signal == "bullish":
#         sl = round(sl_price, symbol_info.digits)
#         tp = round(price + tp_dist, symbol_info.digits)
#     else:
#         sl = round(sl_price, symbol_info.digits)
#         tp = round(price - tp_dist, symbol_info.digits)
    
#     # Validate stops
#     if signal == "bullish":
#         if sl >= price or tp <= price:
#             #print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             #print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     #print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     #print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

#     # Try different filling modes
#     filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
#     for filling_type in filling_modes:
#         request = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": symbol,
#             "volume": lot,
#             "type": order_type,
#             "price": price,
#             "sl": float(sl),
#             "tp": float(tp),
#             "deviation": 20,
#             "magic": 234001,
#             "comment": f"EMA+RSI {signal}",
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": filling_type,
#         }

#         result = mt5.order_send(request)
        
#         # Check if result is None
#         if result is None:
#             error = mt5.last_error()
#             #print(f"✗ Order send failed (returned None) with {filling_type}")
#             #print(f"  MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             #print(f"✓ {signal.upper()} trade placed successfully!")
#             #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             #print(f"  RSI: {curr_candle['rsi']:.2f}")
#             #print(f"  ATR: {current_atr:.5f}")
#             return
#         elif result.retcode == 10018:
#             #print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             #print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     #print(f"✗ Order failed with all filling modes.")


# def is_market_open(symbol: str) -> bool:
#     """Check if market is open and tradable"""
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         return False

#     if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
#         return False

#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             return False

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         return False

#     if tick.bid <= 0 or tick.ask <= 0:
#         return False

#     # Check tick freshness
#     tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
#     now = datetime.now(timezone.utc)

#     if (now - tick_time).total_seconds() > 60:
#         return False

#     # Avoid rollover period
#     if (tick_time.hour == 23 and tick_time.minute >= 55) or \
#        (tick_time.hour == 0 and tick_time.minute <= 5):
#         return False

#     return True


# def check_existing_position(symbol):
#     """Check if there's already an open position for this symbol"""
#     positions = mt5.positions_get(symbol=symbol)
#     if positions is None:
#         return False
#     return len(positions) > 0


# def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5):
#     """Run the EMA crossover strategy with RSI filter"""
    
#     # Check for existing position
#     if check_existing_position(symbol):
#         #print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         #print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         #print(f"[{symbol}] No valid signal")
#         return
    
#     #print(f"[{symbol}] {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     #print(f"  RSI: {curr_candle['rsi']:.2f}")
#     #print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# #print("=" * 60)
# #print("EMA CROSSOVER + RSI FILTER STRATEGY")
# #print("21/51 EMA on 5-Min Chart + RSI(14)")
# #print("Bullish: RSI >= 60 | Bearish: RSI <= 40")
# #print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         #print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     #print(f"[{pair}] Market not open - skipping")
#                     continue
                
#                 # Get current candle time
#                 df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#                 if df is not None:
#                     current_candle_time = df.iloc[-1]['time']
                    
#                     # Only check if new candle formed
#                     if pair not in last_check or last_check[pair] != current_candle_time:
#                         run_ema_strategy(pair, mt5.TIMEFRAME_M5)
#                         last_check[pair] = current_candle_time
                
#             except Exception as e:
#                 #print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         #print("\n" + "-" * 100)
#         time.sleep(60)

# except KeyboardInterrupt:
#     #print("\n\nStopping strategy...")

# finally:
#     shutdown_mt5()



"""
This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range.
"""


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os


# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             #print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo",
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=5044597561,
#             password="Ej@6UjSs",
#             server="MetaQuotes-Demo"
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         #print("✓ Connected to MT5")
#         #print(f"  Account: {account_info.login}")
#         #print(f"  Balance: ${account_info.balance}")
#     else:
#         #print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     #print("MT5 disconnected")


# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=100):
#     """Get candle data for analysis"""
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
#     if rates is None or len(rates) == 0:
#         return None
    
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df


# def calculate_ema(df, period):
#     """Calculate Exponential Moving Average"""
#     return df['close'].ewm(span=period, adjust=False).mean()


# def calculate_rsi(df, period=14):
#     """Calculate Relative Strength Index (RSI)"""
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi


# def detect_ema_crossover(df, rsi_threshold_bullish=60, rsi_threshold_bearish=40):
#     """
#     Detect EMA crossover signals WITH RSI filter
#     - Bullish: 21 EMA crosses above 51 EMA AND RSI >= 60
#     - Bearish: 21 EMA crosses below 51 EMA AND RSI <= 40
#     Returns: 'bullish', 'bearish', or 'neutral'
#     """
#     if len(df) < 52:  # Need at least 52 candles for 51 EMA
#         return "neutral", None, None
    
#     # Calculate EMAs
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
    
#     # Calculate RSI
#     df['rsi'] = calculate_rsi(df, period=14)
    
#     # Get last 3 candles for crossover detection
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
#     prev2 = df.iloc[-3]
    
#     # Check for bullish crossover (21 EMA crosses above 51 EMA) + RSI >= 60
#     if (prev['ema_21'] <= prev['ema_51'] and 
#         curr['ema_21'] > curr['ema_51'] and
#         curr['rsi'] >= rsi_threshold_bullish):
#         return "bullish", curr, df
    
#     # Check for bearish crossover (21 EMA crosses below 51 EMA) + RSI <= 40
#     if (prev['ema_21'] >= prev['ema_51'] and 
#         curr['ema_21'] < curr['ema_51'] and
#         curr['rsi'] <= rsi_threshold_bearish):
#         return "bearish", curr, df
    
#     # Check if crossover occurred but RSI filter not met
#     if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
#         #print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         #print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
#     return "neutral", curr, df


# def place_ema_trade(symbol, signal, curr_candle, df, lot=0.1):
#     """
#     Place trade based on EMA crossover + RSI filter
#     SL: Current candle's high (bearish) or low (bullish)
#     TP: 2x SL distance
#     """
#     if signal not in ["bullish", "bearish"]:
#         #print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         #print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             #print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         #print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate SL based on current candle
#     if signal == "bullish":
#         sl_price = curr_candle['low']
#         sl_dist = abs(price - sl_price)
#     else:
#         sl_price = curr_candle['high']
#         sl_dist = abs(price - sl_price)
    
#     # Set MINIMUM stop distances
#     point = symbol_info.point
    
#     if symbol in ['USDCNH', 'USDZAR', 'USDTRY']:
#         min_pips = 30
#     else:
#         min_pips = 15
    
#     min_distance = min_pips * point * 10
    
#     #print(f"Symbol: {symbol}, Point: {point}")
#     #print(f"Calculated SL dist: {sl_dist:.5f} (from candle {'low' if signal == 'bullish' else 'high'})")
#     #print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     if sl_dist < min_distance:
#         #print("⚠ SL too tight, using minimum distance")
#         sl_dist = min_distance
    
#     # TP is 2x SL distance
#     tp_dist = 2 * sl_dist
    
#     # Calculate final SL/TP
#     if signal == "bullish":
#         sl = round(price - sl_dist, symbol_info.digits)
#         tp = round(price + tp_dist, symbol_info.digits)
#     else:
#         sl = round(price + sl_dist, symbol_info.digits)
#         tp = round(price - tp_dist, symbol_info.digits)
    
#     # Validate stops
#     if signal == "bullish":
#         if sl >= price or tp <= price:
#             #print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             #print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     #print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     #print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

#     # Try different filling modes
#     filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
#     for filling_type in filling_modes:
#         request = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": symbol,
#             "volume": lot,
#             "type": order_type,
#             "price": price,
#             "sl": float(sl),
#             "tp": float(tp),
#             "deviation": 20,
#             "magic": 234001,  # Different magic number for EMA strategy
#             "comment": f"EMA+RSI {signal}",
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": filling_type,
#         }

#         result = mt5.order_send(request)

#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             #print(f"✓ {signal.upper()} trade placed successfully!")
#             #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             #print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             #print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             #print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     #print(f"✗ Order failed with all filling modes. Last: {result.retcode} - {result.comment}")


# def is_market_open(symbol: str) -> bool:
#     """Check if market is open and tradable"""
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         return False

#     if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
#         return False

#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             return False

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         return False

#     if tick.bid <= 0 or tick.ask <= 0:
#         return False

#     # Check tick freshness
#     tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
#     now = datetime.now(timezone.utc)

#     if (now - tick_time).total_seconds() > 60:
#         return False

#     # Avoid rollover period
#     if (tick_time.hour == 23 and tick_time.minute >= 55) or \
#        (tick_time.hour == 0 and tick_time.minute <= 5):
#         return False

#     return True


# def check_existing_position(symbol):
#     """Check if there's already an open position for this symbol"""
#     positions = mt5.positions_get(symbol=symbol)
#     if positions is None:
#         return False
#     return len(positions) > 0


# def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5):
#     """Run the EMA crossover strategy with RSI filter"""
    
#     # Check for existing position
#     if check_existing_position(symbol):
#         #print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         #print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         #print(f"[{symbol}] No valid signal")
#         return
    
#     #print(f"[{symbol}] {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     #print(f"  RSI: {curr_candle['rsi']:.2f}")
#     #print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# #print("=" * 60)
# #print("EMA CROSSOVER + RSI FILTER STRATEGY")
# #print("21/51 EMA on 5-Min Chart + RSI(14)")
# #print("Bullish: RSI >= 60 | Bearish: RSI <= 40")
# #print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         #print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     #print(f"[{pair}] Market not open - skipping")
#                     continue
                
#                 # Get current candle time
#                 df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#                 if df is not None:
#                     current_candle_time = df.iloc[-1]['time']
                    
#                     # Only check if new candle formed
#                     if pair not in last_check or last_check[pair] != current_candle_time:
#                         run_ema_strategy(pair, mt5.TIMEFRAME_M5)
#                         last_check[pair] = current_candle_time
                
#             except Exception as e:
#                 #print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         #print("\n" + "-" * 60)
#         time.sleep(30)

# except KeyboardInterrupt:
#     #print("\n\nStopping strategy...")

# finally:
#     shutdown_mt5()


"""
This below script is for 21 EMA crossover to the 51 EMA 
"""


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os
# "\MetaTrader 5.lnk"

# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             #print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=5044120156,
#             password="ZbMm@6Ib",
#             server="MetaQuotes-Demo",
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=5044120156,
#             password="ZbMm@6Ib",
#             server="MetaQuotes-Demo"
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         #print("✓ Connected to MT5")
#         #print(f"  Account: {account_info.login}")
#         #print(f"  Balance: ${account_info.balance}")
#     else:
#         #print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     #print("MT5 disconnected")


# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=100):
#     """Get candle data for analysis"""
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
#     if rates is None or len(rates) == 0:
#         return None
    
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df


# def calculate_ema(df, period):
#     """Calculate Exponential Moving Average"""
#     return df['close'].ewm(span=period, adjust=False).mean()


# def detect_ema_crossover(df):
#     """
#     Detect EMA crossover signals
#     Returns: 'bullish', 'bearish', or 'neutral'
#     """
#     if len(df) < 52:  # Need at least 52 candles for 51 EMA
#         return "neutral", None, None
    
#     # Calculate EMAs
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
    
#     # Get last 3 candles for crossover detection
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
#     prev2 = df.iloc[-3]
    
#     # Check for bullish crossover (21 EMA crosses above 51 EMA)
#     if (prev['ema_21'] <= prev['ema_51'] and 
#         curr['ema_21'] > curr['ema_51']):
#         return "bullish", curr, df
    
#     # Check for bearish crossover (21 EMA crosses below 51 EMA)
#     if (prev['ema_21'] >= prev['ema_51'] and 
#         curr['ema_21'] < curr['ema_51']):
#         return "bearish", curr, df
    
#     return "neutral", curr, df


# def place_ema_trade(symbol, signal, curr_candle, df, lot=0.1):
#     """
#     Place trade based on EMA crossover
#     SL: Current candle's high (bearish) or low (bullish)
#     TP: 2x SL distance
#     """
#     if signal not in ["bullish", "bearish"]:
#         #print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         #print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             #print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         #print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate SL based on current candle
#     if signal == "bullish":
#         sl_price = curr_candle['low']
#         sl_dist = abs(price - sl_price)
#     else:
#         sl_price = curr_candle['high']
#         sl_dist = abs(price - sl_price)
    
#     # Set MINIMUM stop distances
#     point = symbol_info.point
    
#     if symbol in ['USDCNH', 'USDZAR', 'USDTRY']:
#         min_pips = 30
#     else:
#         min_pips = 15
    
#     min_distance = min_pips * point * 10
    
#     #print(f"Symbol: {symbol}, Point: {point}")
#     #print(f"Calculated SL dist: {sl_dist:.5f} (from candle {'low' if signal == 'bullish' else 'high'})")
#     #print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     if sl_dist < min_distance:
#         #print("⚠ SL too tight, using minimum distance")
#         sl_dist = min_distance
    
#     # TP is 2x SL distance
#     tp_dist = 2 * sl_dist
    
#     # Calculate final SL/TP
#     if signal == "bullish":
#         sl = round(price - sl_dist, symbol_info.digits)
#         tp = round(price + tp_dist, symbol_info.digits)
#     else:
#         sl = round(price + sl_dist, symbol_info.digits)
#         tp = round(price - tp_dist, symbol_info.digits)
    
#     # Validate stops
#     if signal == "bullish":
#         if sl >= price or tp <= price:
#             #print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             #print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     #print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     #print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

#     # Try different filling modes
#     filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
#     for filling_type in filling_modes:
#         request = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": symbol,
#             "volume": lot,
#             "type": order_type,
#             "price": price,
#             "sl": float(sl),
#             "tp": float(tp),
#             "deviation": 20,
#             "magic": 234001,  # Different magic number for EMA strategy
#             "comment": f"EMA {signal}",
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": filling_type,
#         }

#         result = mt5.order_send(request)

#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             #print(f"✓ {signal.upper()} trade placed successfully!")
#             #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             return
#         elif result.retcode == 10018:
#             #print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             #print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     #print(f"✗ Order failed with all filling modes. Last: {result.retcode} - {result.comment}")


# def is_market_open(symbol: str) -> bool:
#     """Check if market is open and tradable"""
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         return False

#     if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
#         return False

#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             return False

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         return False

#     if tick.bid <= 0 or tick.ask <= 0:
#         return False

#     # Check tick freshness
#     tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
#     now = datetime.now(timezone.utc)

#     if (now - tick_time).total_seconds() > 60:
#         return False

#     # Avoid rollover period
#     if (tick_time.hour == 23 and tick_time.minute >= 55) or \
#        (tick_time.hour == 0 and tick_time.minute <= 5):
#         return False

#     return True


# def check_existing_position(symbol):
#     """Check if there's already an open position for this symbol"""
#     positions = mt5.positions_get(symbol=symbol)
#     if positions is None:
#         return False
#     return len(positions) > 0


# def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5):
#     """Run the EMA crossover strategy"""
    
#     # Check for existing position
#     if check_existing_position(symbol):
#         #print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         #print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         #print(f"[{symbol}] No crossover signal")
#         return
    
#     #print(f"[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED!")
#     #print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     #print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     #print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# #print("=" * 60)
# #print("EMA CROSSOVER STRATEGY (21/51 EMA on 5-Min Chart)")
# #print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         #print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     #print(f"[{pair}] Market not open - skipping")
#                     continue
                
#                 # Get current candle time
#                 df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#                 if df is not None:
#                     current_candle_time = df.iloc[-1]['time']
                    
#                     # Only check if new candle formed
#                     if pair not in last_check or last_check[pair] != current_candle_time:
#                         run_ema_strategy(pair, mt5.TIMEFRAME_M5)
#                         last_check[pair] = current_candle_time
                
#             except Exception as e:
#                 #print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         #print("\n" + "-" * 60)
#         time.sleep(30)

# except KeyboardInterrupt:
#     #print("\n\nStopping strategy...")

# finally:
#     shutdown_mt5()