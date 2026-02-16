"""
21 EMA x 51 EMA Crossover Strategy - LIVE TRADING
With 51 EMA Trailing Stop Loss

Strategy Rules:
1. Entry: 21 EMA crosses 51 EMA (bullish/bearish)
2. Stop Loss: Dynamically trails with 51 EMA (updates every minute)
3. Take Profit: NONE - ride the trend until SL is hit
4. Position Management: Update SL every minute to follow 51 EMA
5. Optional Filters: 200 EMA, RSI (M5 + M30)
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import os
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIGURATION =====================

# Account Settings
ACCOUNT_BALANCE = 100
RISK_PER_TRADE = 5  # $5 risk per trade

# Trading Pairs
CURRENCY_PAIRS = ["GBPUSD", "EURUSD", "XAUUSD"]

# Timeframe
TIMEFRAME = mt5.TIMEFRAME_M5

# Strategy Filters (matching backtest config)
REQUIRE_200_EMA = False  # Set to True to require price above/below 200 EMA
REQUIRE_RSI_FILTER = True  # Set to True to enable RSI filter
RSI_BULLISH_THRESHOLD = 60  # RSI must be >= this for bullish trades
RSI_BEARISH_THRESHOLD = 40  # RSI must be <= this for bearish trades
SL_BUFFER_PIPS = 2  # Buffer in pips for SL (prevents premature stops)

# Magic Number (unique identifier for this strategy)
MAGIC_NUMBER = 234003

# Tracking
last_check = {}

# ===================== CONNECTION FUNCTIONS =====================

def connect_mt5(portable=True):
    """Connect to MT5"""
    if portable:
        mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        if not os.path.exists(mt5_path):
            print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
            portable = False
    
    if portable:
        if not mt5.initialize(
            path=mt5_path,
            login=int(os.getenv("MT5_LOGIN")),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            timeout=60000,
            portable=True
        ):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}")
    else:
        if not mt5.initialize(
            login=int(os.getenv("MT5_LOGIN")),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
        ):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}")
    
    account_info = mt5.account_info()
    if account_info:
        print("✓ Connected to MT5")
        print(f"  Account: {account_info.login}")
        print(f"  Balance: ${account_info.balance:.2f}")
        print(f"  Equity: ${account_info.equity:.2f}")
        print(f"  Leverage: 1:{account_info.leverage}")
        return True
    return False


def shutdown_mt5():
    """Disconnect from MT5"""
    mt5.shutdown()
    print("MT5 disconnected")


# ===================== INDICATOR FUNCTIONS =====================

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
        return 10  # Default fallback
    
    point = symbol_info.point
    
    if "JPY" in symbol:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    
    # For USD pairs
    if symbol.endswith("USD"):
        pip_value_per_lot = pip_size * symbol_info.trade_contract_size
    elif symbol.startswith("USD"):
        pip_value_per_lot = (pip_size / price) * symbol_info.trade_contract_size
    else:
        pip_value_per_lot = (pip_size / price) * symbol_info.trade_contract_size
    
    return pip_value_per_lot


# ===================== DATA FUNCTIONS =====================

def get_candle_data(symbol, timeframe, n=250):
    """Get candle data for analysis"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_indicators(df, include_rsi=False):
    """Calculate all indicators (21 EMA, 51 EMA, 200 EMA, optional RSI)"""
    df['ema_21'] = ema(df['close'], 21)
    df['ema_51'] = ema(df['close'], 51)
    df['ema_200'] = ema(df['close'], 200)
    
    if include_rsi:
        df['rsi'] = rsi(df['close'], 14)
    
    return df


def get_m30_rsi(symbol):
    """Get M30 RSI value"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 50)
    if rates is None or len(rates) == 0:
        return None
    
    df_m30 = pd.DataFrame(rates)
    df_m30['time'] = pd.to_datetime(df_m30['time'], unit='s')
    df_m30['rsi'] = rsi(df_m30['close'], 14)
    
    return df_m30.iloc[-1]['rsi']


# ===================== SIGNAL DETECTION =====================

def detect_ema_crossover(symbol):
    """
    Detect EMA crossover signals with optional filters
    Returns: signal ('bullish', 'bearish', 'neutral'), current candle data, dataframe
    """
    # Get M5 data
    df = get_candle_data(symbol, TIMEFRAME, n=250)
    if df is None or len(df) < 201:
        return "neutral", None, None
    
    # Calculate indicators
    df = calculate_indicators(df, include_rsi=REQUIRE_RSI_FILTER)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Detect crossovers
    bullish = (
        prev['ema_21'] <= prev['ema_51'] and
        curr['ema_21'] > curr['ema_51']
    )
    
    bearish = (
        prev['ema_21'] >= prev['ema_51'] and
        curr['ema_21'] < curr['ema_51']
    )
    
    # Apply 200 EMA filter if enabled
    if REQUIRE_200_EMA:
        if bullish:
            bullish = curr['ema_21'] > curr['ema_200'] and curr['ema_51'] > curr['ema_200']
            if not bullish:
                print(f"  ℹ [{symbol}] Bullish crossover rejected: below 200 EMA")
        
        if bearish:
            bearish = curr['ema_21'] < curr['ema_200'] and curr['ema_51'] < curr['ema_200']
            if not bearish:
                print(f"  ℹ [{symbol}] Bearish crossover rejected: above 200 EMA")
    
    # Apply RSI filter if enabled
    if REQUIRE_RSI_FILTER and (bullish or bearish):
        m5_rsi = curr['rsi']
        m30_rsi = get_m30_rsi(symbol)
        
        if pd.isna(m5_rsi) or m30_rsi is None or pd.isna(m30_rsi):
            print(f"  ⚠ [{symbol}] RSI data unavailable, skipping trade")
            bullish = False
            bearish = False
        else:
            if bullish:
                if m30_rsi >= RSI_BULLISH_THRESHOLD:
                # if m5_rsi >= RSI_BULLISH_THRESHOLD and m30_rsi >= RSI_BULLISH_THRESHOLD:
                    print(f"  ✓ [{symbol}] RSI Filter PASSED - M5: {m5_rsi:.1f}, M30: {m30_rsi:.1f}")
                else:
                    print(f"  ✗ [{symbol}] RSI Filter FAILED - M5: {m5_rsi:.1f}, M30: {m30_rsi:.1f} (need >= {RSI_BULLISH_THRESHOLD})")
                    bullish = False
            
            if bearish:
                if m30_rsi <= RSI_BEARISH_THRESHOLD:
                # if m5_rsi <= RSI_BEARISH_THRESHOLD and m30_rsi <= RSI_BEARISH_THRESHOLD:
                    print(f"  ✓ [{symbol}] RSI Filter PASSED - M5: {m5_rsi:.1f}, M30: {m30_rsi:.1f}")
                else:
                    print(f"  ✗ [{symbol}] RSI Filter FAILED - M5: {m5_rsi:.1f}, M30: {m30_rsi:.1f} (need <= {RSI_BEARISH_THRESHOLD})")
                    bearish = False
    
    if bullish:
        return "bullish", curr, df
    elif bearish:
        return "bearish", curr, df
    else:
        return "neutral", curr, df


# ===================== POSITION MANAGEMENT =====================

def get_current_51_ema(symbol):
    """Get current 51 EMA value"""
    df = get_candle_data(symbol, TIMEFRAME, n=100)
    if df is None or len(df) < 51:
        return None
    
    df = calculate_indicators(df, include_rsi=False)
    return df.iloc[-1]['ema_51']


def update_trailing_stop(position):
    """
    Update position's stop loss to current 51 EMA value
    Only updates if it's favorable (moves SL in profit direction)
    """
    symbol = position.symbol
    ticket = position.ticket
    position_type = position.type
    current_sl = position.sl
    entry_price = position.price_open
    
    # Get current 51 EMA
    current_51_ema = get_current_51_ema(symbol)
    if current_51_ema is None:
        print(f"  ⚠ [{symbol}] Could not calculate 51 EMA")
        return False
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False
    
    # Add buffer
    point = symbol_info.point
    pip_multiplier = 10 if "JPY" not in symbol else 1
    buffer = SL_BUFFER_PIPS * point * pip_multiplier
    
    # Calculate new SL with buffer
    if position_type == mt5.ORDER_TYPE_BUY:
        # For buy positions, SL should be below 51 EMA
        new_sl = current_51_ema - buffer
        
        # Only update if new SL is higher than current SL (trailing up)
        if new_sl > current_sl:
            new_sl = round(new_sl, symbol_info.digits)
        else:
            return False  # Don't update
            
    else:  # SELL position
        # For sell positions, SL should be above 51 EMA
        new_sl = current_51_ema + buffer
        
        # Only update if new SL is lower than current SL (trailing down)
        if new_sl < current_sl:
            new_sl = round(new_sl, symbol_info.digits)
        else:
            return False  # Don't update
    
    # Modify position
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": float(new_sl),
        "tp": position.tp,
        "magic": MAGIC_NUMBER,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        print(f"  ✗ [{symbol}] Failed to update SL: {error}")
        return False
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        direction = "↑" if position_type == mt5.ORDER_TYPE_BUY else "↓"
        pips_moved = abs(new_sl - current_sl) / (point * pip_multiplier)
        print(f"  ✓ [{symbol}] SL updated: {current_sl:.5f} → {new_sl:.5f} {direction} (+{pips_moved:.1f} pips)")
        return True
    else:
        print(f"  ✗ [{symbol}] Failed to update SL: {result.retcode} - {result.comment}")
        return False


def update_all_trailing_stops():
    """Update trailing stops for all open positions"""
    positions = mt5.positions_get()
    
    if positions is None or len(positions) == 0:
        return
    
    # Filter positions by our magic number
    our_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
    
    if len(our_positions) == 0:
        return
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔄 Updating trailing stops ({len(our_positions)} position(s))...")
    
    for position in our_positions:
        update_trailing_stop(position)


# ===================== TRADE EXECUTION =====================

def place_ema_trade(symbol, signal, curr_candle):
    """
    Place trade based on EMA crossover
    SL: Initial SL at 51 EMA (will be updated by trailing stop)
    TP: No TP - ride the trend until SL is hit
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"  ✗ [{symbol}] Failed to get symbol info")
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"  ✗ [{symbol}] Failed to add to Market Watch")
            return False
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"  ✗ [{symbol}] Failed to get tick data")
        return False
    
    # Entry price
    price = tick.ask if signal == "bullish" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL
    
    # Initial SL at 51 EMA
    ema_51 = curr_candle['ema_51']
    
    # Add buffer to SL
    point = symbol_info.point
    pip_multiplier = 10 if "JPY" not in symbol else 1
    buffer = SL_BUFFER_PIPS * point * pip_multiplier
    
    if signal == "bullish":
        sl_price = ema_51 - buffer
    else:
        sl_price = ema_51 + buffer
    
    sl = round(sl_price, symbol_info.digits)
    
    # Validate SL
    if signal == "bullish":
        if sl >= price:
            print(f"  ✗ [{symbol}] Invalid SL for bullish trade (SL >= Entry)")
            return False
    else:
        if sl <= price:
            print(f"  ✗ [{symbol}] Invalid SL for bearish trade (SL <= Entry)")
            return False
    
    # Calculate position size based on risk
    sl_dist_price = abs(price - sl)
    sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
    pip_value = calculate_pip_value(symbol, price)
    
    if sl_dist_pips > 0 and pip_value > 0:
        lot_size = RISK_PER_TRADE / (sl_dist_pips * pip_value)
        lot_size = round(lot_size, 2)  # Round to 2 decimals
        
        # Apply min/max limits
        if lot_size < symbol_info.volume_min:
            lot_size = symbol_info.volume_min
        elif lot_size > symbol_info.volume_max:
            lot_size = symbol_info.volume_max
    else:
        print(f"  ✗ [{symbol}] Invalid calculation: SL pips={sl_dist_pips:.1f}, pip_value={pip_value:.2f}")
        return False
    
    # Calculate actual risk
    actual_risk = lot_size * sl_dist_pips * pip_value
    
    # Display trade setup
    print(f"\n{'='*70}")
    print(f"📊 TRADE SETUP - {symbol}")
    print(f"{'='*70}")
    print(f"Signal: {signal.upper()}")
    print(f"Entry Price: {price:.5f}")
    print(f"Initial Stop Loss: {sl:.5f} (51 EMA: {ema_51:.5f})")
    print(f"Take Profit: NONE (Ride the trend)")
    print(f"")
    print(f"SL Distance: {sl_dist_pips:.1f} pips")
    print(f"Position Size: {lot_size:.2f} lots")
    print(f"Risk Amount: ${actual_risk:.2f}")
    print(f"")
    print(f"21 EMA: {curr_candle['ema_21']:.5f}")
    print(f"51 EMA: {curr_candle['ema_51']:.5f}")
    if REQUIRE_200_EMA:
        print(f"200 EMA: {curr_candle['ema_200']:.5f}")
    if REQUIRE_RSI_FILTER:
        print(f"RSI (M5): {curr_candle['rsi']:.1f}")
    print(f"")
    print(f"NOTE: SL will trail the 51 EMA automatically")
    print(f"{'='*70}\n")
    
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
            "tp": 0.0,  # No take profit
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"EMA Trail {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            print(f"  ✗ [{symbol}] Order send failed with {filling_type}: {error}")
            continue
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ [{symbol}] {signal.upper()} trade placed successfully!")
            print(f"  Order Ticket: {result.order}")
            print(f"  Position Ticket: {result.deal if hasattr(result, 'deal') else 'N/A'}")
            return True
        elif result.retcode == 10018:
            print(f"  ✗ [{symbol}] Market is closed")
            return False
        elif result.retcode != 10030:
            print(f"  ✗ [{symbol}] Order failed: {result.retcode} - {result.comment}")
            if filling_type == filling_modes[-1]:
                return False
    
    print(f"  ✗ [{symbol}] Order failed with all filling modes")
    return False


# ===================== MARKET CHECKS =====================

def is_market_open(symbol):
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
    
    tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    
    if (now - tick_time).total_seconds() > 60:
        return False
    
    # Avoid rollover period (23:55 - 00:05 UTC)
    if (tick_time.hour == 23 and tick_time.minute >= 55) or \
       (tick_time.hour == 0 and tick_time.minute <= 5):
        return False
    
    return True


def check_existing_position(symbol):
    """Check if there's already an open position for this symbol"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    
    # Check if any position is from our strategy
    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            return True
    
    return False


# ===================== STRATEGY EXECUTION =====================

def run_strategy_for_symbol(symbol):
    """Run the EMA crossover strategy for a single symbol"""
    try:
        # Check if market is open
        if not is_market_open(symbol):
            return
        
        # Check if we already have a position
        if check_existing_position(symbol):
            return
        
        # Check for new candle (avoid duplicate signals)
        df = get_candle_data(symbol, TIMEFRAME, n=2)
        if df is None:
            return
        
        current_candle_time = df.iloc[-1]['time']
        
        # Only analyze if this is a new candle
        if symbol in last_check and last_check[symbol] == current_candle_time:
            return
        
        last_check[symbol] = current_candle_time
        
        # Detect signal
        signal, curr_candle, df_with_emas = detect_ema_crossover(symbol)
        
        if signal == "neutral":
            return
        
        # Signal detected!
        print(f"\n🎯 [{symbol}] {signal.upper()} CROSSOVER DETECTED!")
        print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
        print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
        
        # Place trade
        place_ema_trade(symbol, signal, curr_candle)
        
    except Exception as e:
        print(f"  ✗ [{symbol}] Error: {e}")


def scan_markets():
    """Scan all currency pairs for trading signals"""
    current_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Scanning markets...")
    print(f"{'='*70}")
    
    # Scan for new signals
    for pair in CURRENCY_PAIRS:
        run_strategy_for_symbol(pair)
    
    # Update trailing stops for existing positions
    update_all_trailing_stops()
    
    print(f"\n{'='*70}")
    print(f"Scan complete. Next scan in 1 minute.")
    print(f"{'='*70}\n")


# ===================== MAIN EXECUTION =====================

def main():
    """Main execution function"""
    print("=" * 70)
    print("21/51 EMA CROSSOVER STRATEGY - LIVE TRADING")
    print("WITH 51 EMA TRAILING STOP LOSS")
    print("=" * 70)
    print(f"Account Balance: ${ACCOUNT_BALANCE}")
    print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
    print(f"Timeframe: M5 (5-Minute Chart)")
    print(f"")
    print(f"Strategy Configuration:")
    print(f"  Entry: 21 EMA crosses 51 EMA")
    print(f"  Stop Loss: 51 EMA (Trailing)")
    print(f"  Take Profit: NONE - Ride the trend")
    print(f"  SL Buffer: {SL_BUFFER_PIPS} pips")
    print(f"  200 EMA Filter: {'ENABLED' if REQUIRE_200_EMA else 'DISABLED'}")
    print(f"  RSI Filter: {'ENABLED' if REQUIRE_RSI_FILTER else 'DISABLED'}")
    if REQUIRE_RSI_FILTER:
        print(f"    - Bullish RSI: >= {RSI_BULLISH_THRESHOLD}")
        print(f"    - Bearish RSI: <= {RSI_BEARISH_THRESHOLD}")
        print(f"    - M30 RSI: Required")
    print(f"")
    print(f"Trading Pairs: {', '.join(CURRENCY_PAIRS)}")
    print(f"Magic Number: {MAGIC_NUMBER}")
    print("=" * 70)
    
    # Connect to MT5
    if not connect_mt5(portable=True):
        print("\n❌ Failed to connect to MT5")
        return
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Run every minute to check for signals AND update trailing stops
    scheduler.add_job(
        scan_markets,
        trigger='cron',
        minute='*',
        second='30',
        id='market_scanner',
        name='Scan markets and update trailing stops',
        max_instances=1
    )
    
    print("\n✓ Strategy started successfully!")
    print("  • Scans for EMA crossover signals every minute")
    print("  • Updates all trailing stops to 51 EMA every minute")
    print("  • Press Ctrl+C to stop\n")
    
    try:
        # Run initial scan
        scan_markets()
        
        # Start scheduler
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\n⚠ Stopping strategy...")
        scheduler.shutdown()
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        shutdown_mt5()
        print("✓ Strategy stopped")


if __name__ == "__main__":
    main()





"""
21 EMA x 51 EMA Crossover Strategy with 51 EMA Trailing Stop Loss

Strategy Rules:
1. Entry: 21 EMA crosses 51 EMA (bullish/bearish)
2. Stop Loss: Dynamically updated to 51 EMA value (trailing)
3. Take Profit: Fixed R:R ratio (default 1:2)
4. Position Management: Update SL every minute to follow 51 EMA
"""

# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.triggers.cron import CronTrigger
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # ===================== CONNECTION FUNCTIONS =====================

# def connect_mt5(portable=True):
#     """Connect to MT5"""
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
#         if not os.path.exists(mt5_path):
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance:.2f}")
#         print(f"  Leverage: 1:{account_info.leverage}")


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


# # ===================== DATA FUNCTIONS =====================

# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=250):
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


# def calculate_indicators(df):
#     """Calculate all indicators (21 EMA, 51 EMA, 200 EMA)"""
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
#     df['ema_200'] = calculate_ema(df, 200)
#     return df


# # ===================== SIGNAL DETECTION =====================

# def detect_ema_crossover(df, require_200ema=True):
#     """
#     Detect EMA crossover signals
#     - Bullish: 21 EMA crosses above 51 EMA (and above 200 EMA if required)
#     - Bearish: 21 EMA crosses below 51 EMA (and below 200 EMA if required)
#     Returns: 'bullish', 'bearish', or 'neutral'
#     """
#     if len(df) < 201:
#         return "neutral", None, None
    
#     df = calculate_indicators(df)
    
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
    
#     # Bullish crossover
#     if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
#         if require_200ema:
#             if curr['ema_21'] > curr['ema_200'] and curr['ema_51'] > curr['ema_200']:
#                 return "bullish", curr, df
#             else:
#                 print(f"ℹ Bullish crossover rejected: below 200 EMA")
#         else:
#             return "bullish", curr, df
    
#     # Bearish crossover
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         if require_200ema:
#             if curr['ema_21'] < curr['ema_200'] and curr['ema_51'] < curr['ema_200']:
#                 return "bearish", curr, df
#             else:
#                 print(f"ℹ Bearish crossover rejected: above 200 EMA")
#         else:
#             return "bearish", curr, df
    
#     return "neutral", curr, df


# # ===================== POSITION MANAGEMENT =====================

# def get_current_51_ema(symbol, timeframe=mt5.TIMEFRAME_M5):
#     """Get current 51 EMA value"""
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 51:
#         return None
    
#     df = calculate_indicators(df)
#     return df.iloc[-1]['ema_51']


# def update_trailing_stop(position, current_51_ema):
#     """
#     Update position's stop loss to current 51 EMA value
#     Only updates if it's favorable (moves SL in profit direction)
#     """
#     symbol = position.symbol
#     ticket = position.ticket
#     position_type = position.type
#     current_sl = position.sl
#     entry_price = position.price_open
    
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         return False
    
#     # Add small buffer to avoid hitting SL on minor fluctuations
#     buffer_pips = 2  # 2 pips buffer
#     point = symbol_info.point
#     pip_multiplier = 10 if "JPY" not in symbol else 1
#     buffer = buffer_pips * point * pip_multiplier
    
#     # Calculate new SL with buffer
#     if position_type == mt5.ORDER_TYPE_BUY:
#         # For buy positions, SL should be below 51 EMA
#         new_sl = current_51_ema - buffer
        
#         # Only update if new SL is higher than current SL (trailing up)
#         # and still below entry price (don't move SL above breakeven yet)
#         if new_sl > current_sl and new_sl < entry_price:
#             new_sl = round(new_sl, symbol_info.digits)
#         elif new_sl >= entry_price:
#             # If 51 EMA crossed above entry, set SL to breakeven
#             new_sl = round(entry_price, symbol_info.digits)
#         else:
#             return False  # Don't update
            
#     else:  # SELL position
#         # For sell positions, SL should be above 51 EMA
#         new_sl = current_51_ema + buffer
        
#         # Only update if new SL is lower than current SL (trailing down)
#         # and still above entry price
#         if new_sl < current_sl and new_sl > entry_price:
#             new_sl = round(new_sl, symbol_info.digits)
#         elif new_sl <= entry_price:
#             # If 51 EMA crossed below entry, set SL to breakeven
#             new_sl = round(entry_price, symbol_info.digits)
#         else:
#             return False  # Don't update
    
#     # Modify position
#     request = {
#         "action": mt5.TRADE_ACTION_SLTP,
#         "position": ticket,
#         "symbol": symbol,
#         "sl": float(new_sl),
#         "tp": position.tp,
#         "magic": 234002,
#     }
    
#     result = mt5.order_send(request)
    
#     if result is None:
#         error = mt5.last_error()
#         print(f"  ✗ Failed to update SL: {error}")
#         return False
    
#     if result.retcode == mt5.TRADE_RETCODE_DONE:
#         direction = "↑" if position_type == mt5.ORDER_TYPE_BUY else "↓"
#         print(f"  ✓ {symbol} SL updated: {current_sl:.5f} → {new_sl:.5f} {direction}")
#         return True
#     else:
#         print(f"  ✗ Failed to update SL: {result.retcode} - {result.comment}")
#         return False


# def update_all_trailing_stops():
#     """Update trailing stops for all open positions"""
#     positions = mt5.positions_get()
    
#     if positions is None or len(positions) == 0:
#         return
    
#     print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating trailing stops ({len(positions)} positions)...")
    
#     for position in positions:
#         # Only manage positions opened by this strategy
#         if position.magic != 234002:
#             continue
        
#         symbol = position.symbol
        
#         # Get current 51 EMA
#         current_51_ema = get_current_51_ema(symbol, mt5.TIMEFRAME_M5)
        
#         if current_51_ema is None:
#             print(f"  ⚠ {symbol}: Could not calculate 51 EMA")
#             continue
        
#         # Update trailing stop
#         update_trailing_stop(position, current_51_ema)


# # ===================== TRADE EXECUTION =====================

# def place_ema_trade(symbol, signal, curr_candle, df, risk_amount=5.0):
#     """
#     Place trade based on EMA crossover
#     SL: Initial SL at 51 EMA (will be updated by trailing stop)
#     TP: No TP - ride the trend until SL is hit
#     """
#     if signal not in ["bullish", "bearish"]:
#         print("No trade signal - skipping")
#         return

#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Initial SL at 51 EMA
#     ema_51 = curr_candle['ema_51']
    
#     # Add small buffer to SL
#     point = symbol_info.point
#     pip_multiplier = 10 if "JPY" not in symbol else 1
#     buffer = 2 * point * pip_multiplier  # 2 pip buffer
    
#     if signal == "bullish":
#         sl_price = ema_51 - buffer
#     else:
#         sl_price = ema_51 + buffer
    
#     sl = round(sl_price, symbol_info.digits)
    
#     # Validate SL
#     if signal == "bullish":
#         if sl >= price:
#             print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
#             return
#     else:
#         if sl <= price:
#             print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
#             return
    
#     sl_dist_price = abs(price - sl)
    
#     # Get contract size and calculate position size
#     contract_size = symbol_info.trade_contract_size
    
#     # Calculate pip size and value
#     if "JPY" in symbol:
#         pip_size = 0.01
#     else:
#         pip_size = 0.0001
    
#     sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
#     # Calculate pip value for 1 lot
#     if symbol.startswith("USD"):
#         pip_value_per_lot = (pip_size / price) * contract_size
#     elif symbol.endswith("USD"):
#         pip_value_per_lot = pip_size * contract_size
#     else:
#         pip_value_per_lot = (pip_size / price) * contract_size
    
#     # Calculate lot size based on risk
#     if sl_dist_pips > 0 and pip_value_per_lot > 0:
#         lot_size = risk_amount / (sl_dist_pips * pip_value_per_lot)
#     else:
#         print(f"⚠ Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
#         return
    
#     # Round to volume step
#     volume_step = symbol_info.volume_step
#     lot_size = round(lot_size / volume_step) * volume_step
    
#     # Apply min/max limits
#     if lot_size < symbol_info.volume_min:
#         lot_size = symbol_info.volume_min
#     elif lot_size > symbol_info.volume_max:
#         lot_size = symbol_info.volume_max
    
#     # NO TAKE PROFIT - ride the trend until SL is hit
#     tp = 0.0
    
#     # Calculate P&L expectations
#     actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
    
#     print(f"\n{'='*70}")
#     print(f"TRADE SETUP - {symbol} (51 EMA TRAILING STOP)")
#     print(f"{'='*70}")
#     print(f"Signal: {signal.upper()}")
#     print(f"Entry Price: {price:.5f}")
#     print(f"Initial Stop Loss: {sl:.5f} (51 EMA: {ema_51:.5f})")
#     print(f"Take Profit: NONE (Ride the trend)")
#     print(f"")
#     print(f"SL Distance: {sl_dist_pips:.1f} pips")
#     print(f"")
#     print(f"Position Size: {lot_size:.2f} lots")
#     print(f"Expected RISK: ${actual_risk:.2f}")
#     print(f"Expected REWARD: UNLIMITED (trails with 51 EMA)")
#     print(f"")
#     print(f"21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"NOTE: SL will trail the 51 EMA automatically")
#     print(f"{'='*70}\n")

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
#             "tp": 0.0,  # No take profit
#             "deviation": 20,
#             "magic": 234002,  # Different magic number for this strategy
#             "comment": f"EMA Trail {signal}",
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": filling_type,
#         }

#         result = mt5.order_send(request)
        
#         if result is None:
#             error = mt5.last_error()
#             print(f"✗ Order send failed with {filling_type}: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  Order Ticket: {result.order}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes")


# # ===================== MARKET CHECKS =====================

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

#     tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
#     now = datetime.now(timezone.utc)

#     if (now - tick_time).total_seconds() > 60:
#         return False

#     # Avoid rollover
#     if (tick_time.hour == 23 and tick_time.minute >= 55) or \
#        (tick_time.hour == 0 and tick_time.minute <= 5):
#         return False

#     return True


# def check_existing_position(symbol):
#     """Check if there's already an open position for this symbol"""
#     positions = mt5.positions_get(symbol=symbol)
#     if positions is None:
#         return False
    
#     # Check if any position is from our strategy (magic number 234002)
#     for pos in positions:
#         if pos.magic == 234002:
#             return True
    
#     return False


# # ===================== STRATEGY EXECUTION =====================

# def run_ema_strategy(symbol, timeframe=mt5.TIMEFRAME_M5, risk_per_trade=5.0):
#     """Run the EMA crossover strategy with trailing stop"""
    
#     if check_existing_position(symbol):
#         # print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     df = get_candle_data(symbol, timeframe, n=250)
#     if df is None or len(df) < 201:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df, require_200ema=False)
    
#     if signal == "neutral":
#         # print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"\n[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
    
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas, 
#                    risk_amount=risk_per_trade)


# def scan_markets():
#     """Scan all currency pairs for trading signals"""
#     current_time = datetime.now()
#     print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
    
#     for pair in CURRENCY_PAIRS:
#         try:
#             if not is_market_open(pair):
#                 # print(f"[{pair}] Market not open - skipping")
#                 continue
            
#             # Check for new candle
#             df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#             if df is not None:
#                 current_candle_time = df.iloc[-1]['time']
                
#                 if pair not in last_check or last_check[pair] != current_candle_time:
#                     run_ema_strategy(pair, mt5.TIMEFRAME_M5, 
#                                    risk_per_trade=RISK_PER_TRADE)
#                     last_check[pair] = current_candle_time
            
#         except Exception as e:
#             print(f"[{pair}] Error: {e}")
#             continue
    
#     # Update trailing stops for all positions
#     update_all_trailing_stops()
    
#     print("-" * 100)


# # ===================== MAIN EXECUTION =====================

# # Configuration
# ACCOUNT_BALANCE = 100
# RISK_PER_TRADE = 5      # $5 risk per trade

# CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
# last_check = {}

# print("=" * 70)
# print("21/51 EMA CROSSOVER STRATEGY WITH 51 EMA TRAILING STOP")
# print("=" * 70)
# print(f"Account Balance: ${ACCOUNT_BALANCE}")
# print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
# print(f"Timeframe: 5-Minute Chart")
# print(f"Entry: 21 EMA crosses 51 EMA")
# print(f"Stop Loss: 51 EMA (Trailing - Updates Every Minute)")
# print(f"Take Profit: NONE - Ride the trend until stopped out")
# print("=" * 70)

# connect_mt5(portable=True)

# # Create scheduler
# scheduler = BlockingScheduler()

# # Run every minute to check for signals AND update trailing stops
# scheduler.add_job(
#     scan_markets,
#     trigger='cron',
#     minute='*',
#     second='5',
#     id='market_scanner',
#     name='Scan markets and update trailing stops',
#     max_instances=1
# )

# print("\n✓ Scheduler initialized")
# print("  • Scans for EMA crossover signals every minute")
# print("  • Updates all trailing stops to 51 EMA every minute")
# print("  Press Ctrl+C to stop\n")

# try:
#     scheduler.start()
# except (KeyboardInterrupt, SystemExit):
#     print("\n\nStopping strategy...")
#     scheduler.shutdown()
# finally:
#     shutdown_mt5()











"""
This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range. And the stop loss is 1.5 ATR 
with position sizing based on risk amount ($5 risk, $10 profit per trade).
1. SL at 20 EMA +- 1 ATR
2. EMA cross should be above the 200 EMA only.
"""


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os
# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.triggers.cron import CronTrigger
# import os
# from dotenv import load_dotenv

# load_dotenv()  # loads .env file

# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         # mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\MetaTrader 5.lnk"
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance}")
#         print(f"  Leverage: 1:{account_info.leverage}")
#         pass
#     else:
#         print("✓ Connected to MT5")
#         pass


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


# def get_candle_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, n=250):
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
#     if len(df) < 201:  # Need at least 52 candles for 51 EMA
#         return "neutral", None, None
    
#     # Calculate EMAs
#     df['ema_21'] = calculate_ema(df, 21)
#     df['ema_51'] = calculate_ema(df, 51)
#     df['ema_200'] = calculate_ema(df, 200)

    
#     # Calculate RSI
#     df['rsi'] = calculate_rsi(df, period=14)
    
#     # Get last 3 candles for crossover detection
#     curr = df.iloc[-1]
#     prev = df.iloc[-2]
#     prev2 = df.iloc[-3]
    
#     # Bullish crossover + RSI + above 200 EMA
#     if (
#         prev['ema_21'] <= prev['ema_51'] and
#         curr['ema_21'] > curr['ema_51'] and
#         curr['rsi'] >= rsi_threshold_bullish and
#         curr['ema_21'] > curr['ema_200'] and
#         curr['ema_51'] > curr['ema_200']
#     ):
#         return "bullish", curr, df
        
#     # Bearish crossover + RSI + below 200 EMA
#     if (
#         prev['ema_21'] >= prev['ema_51'] and
#         curr['ema_21'] < curr['ema_51'] and
#         curr['rsi'] <= rsi_threshold_bearish and
#         curr['ema_21'] < curr['ema_200'] and
#         curr['ema_51'] < curr['ema_200']
#     ):
#         return "bearish", curr, df
    
#     if prev['ema_21'] <= prev['ema_51'] and curr['ema_21'] > curr['ema_51']:
#         if curr['rsi'] < rsi_threshold_bullish:
#             print(f"ℹ Bullish crossover rejected: RSI {curr['rsi']:.2f}")
#         elif not (curr['ema_21'] > curr['ema_200'] and curr['ema_51'] > curr['ema_200']):
#             print("ℹ Bullish crossover rejected: below 200 EMA")

#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         if curr['rsi'] > rsi_threshold_bearish:
#             print(f"ℹ Bearish crossover rejected: RSI {curr['rsi']:.2f}")
#         elif not (curr['ema_21'] < curr['ema_200'] and curr['ema_51'] < curr['ema_200']):
#             print("ℹ Bearish crossover rejected: above 200 EMA")
    
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
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         print(f"⚠ Invalid ATR value: {current_atr}, cannot place trade")
#         return
    
#     # SL based on 21 EMA ± 1 × ATR
#     atr_buffer = 1.0 * current_atr

#     ema_21 = curr_candle['ema_21']

#     if signal == "bullish":
#         sl_price = ema_21 - atr_buffer
#     else:
#         sl_price = ema_21 + atr_buffer

#     sl_dist_price = abs(price - sl_price)
    
#     # Round SL to proper digits
#     sl = round(sl_price, symbol_info.digits)
    
#     # Validate SL
#     if signal == "bullish":
#         if sl >= price:
#             print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
#             return
#     else:
#         if sl <= price:
#             print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
#             return
    
#     # Get contract size and point value
#     contract_size = symbol_info.trade_contract_size
#     point = symbol_info.point
    
#     # Calculate pip value per lot
#     # For forex pairs, 1 pip = 10 points (except JPY pairs where 1 pip = 100 points)
#     if "JPY" in symbol:
#         pip_multiplier = 10
#         pip_size = 0.01
#     else:
#         pip_multiplier = 10
#         pip_size = 0.0001
    
#     # SL distance in pips
#     sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
#     # Calculate pip value for 1 standard lot (100,000 units)
#     # For USD-based pairs or when account currency is USD
#     if symbol.startswith("USD"):
#         # USD is base - divide by price
#         pip_value_per_lot = (pip_size / price) * contract_size
#     elif symbol.endswith("USD"):
#         # USD is quote - multiply directly
#         pip_value_per_lot = pip_size * contract_size
#     else:
#         # Cross pairs
#         pip_value_per_lot = (pip_size / price) * contract_size
    
#     # Calculate required lot size for desired risk
#     # risk_amount = lot_size × sl_dist_pips × pip_value_per_lot
#     if sl_dist_pips > 0 and pip_value_per_lot > 0:
#         lot_size = risk_amount / (sl_dist_pips * pip_value_per_lot)
#     else:
#         print(f"⚠ Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
#         return
    
#     # Round to broker's volume step (usually 0.01)
#     volume_step = symbol_info.volume_step
#     lot_size = round(lot_size / volume_step) * volume_step
    
#     # Apply min/max lot size limits
#     if lot_size < symbol_info.volume_min:
#         print(f"⚠ Calculated lot size {lot_size:.2f} is below minimum {symbol_info.volume_min}")
#         lot_size = symbol_info.volume_min
#     elif lot_size > symbol_info.volume_max:
#         print(f"⚠ Calculated lot size {lot_size:.2f} exceeds maximum {symbol_info.volume_max}")
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
#             print("ERROR: Invalid take profit for bullish trade")
#             return
#     else:
#         if tp >= price:
#             print("ERROR: Invalid take profit for bearish trade")
#             return
    
#     # Calculate actual risk/reward in dollars
#     actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
#     actual_reward = lot_size * tp_dist_pips * pip_value_per_lot
    
#     print(f"\n{'='*70}")
#     print(f"TRADE SETUP - {symbol}")
#     print(f"{'='*70}")
#     print(f"Signal: {signal.upper()}")
#     print(f"Entry Price: {price:.5f}")
#     print(f"Stop Loss: {sl:.5f} (Distance: {sl_dist_pips:.1f} pips)")
#     print(f"Take Profit: {tp:.5f} (Distance: {tp_dist_pips:.1f} pips)")
#     print(f"")
#     print(f"Position Size: {lot_size:.2f} lots")
#     print(f"Contract Size: {contract_size:,.0f}")
#     print(f"Pip Value/Lot: ${pip_value_per_lot:.2f}")
#     print(f"")
#     print(f"Expected RISK: ${actual_risk:.2f} (Target: ${risk_amount:.2f})")
#     print(f"Expected REWARD: ${actual_reward:.2f} (Target: ${reward_amount:.2f})")
#     print(f"Risk/Reward Ratio: 1:{actual_reward/actual_risk:.2f}")
#     print(f"")
#     print(f"ATR(14): {current_atr:.5f} | ATR Buffer (1.5×): {atr_buffer:.5f}")
#     print(f"{'='*70}\n")

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
#             print(f"✗ Order send failed (returned None) with {filling_type}")
#             print(f"  MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  Order Ticket: {result.order}")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes.")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=250)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"\n[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  RSI: {curr_candle['rsi']:.2f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#     f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade with risk-based position sizing
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas, 
#                    risk_amount=risk_per_trade, reward_amount=reward_per_trade)


# def scan_markets():
#     """
#     Scan all currency pairs for trading signals.
#     This function is called by the scheduler.
#     """
#     current_time = datetime.now()
#     print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
    
#     for pair in currency_pair_list:
#         try:
#             if not is_market_open(pair):
#                 print(f"[{pair}] Market not open - skipping")
#                 continue
            
#             # Get current candle time
#             df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#             if df is not None:
#                 current_candle_time = df.iloc[-1]['time']
                
#                 # Only check if new candle formed
#                 if pair not in last_check or last_check[pair] != current_candle_time:
#                     run_ema_strategy(pair, mt5.TIMEFRAME_M5, 
#                                    risk_per_trade=RISK_PER_TRADE, 
#                                    reward_per_trade=REWARD_PER_TRADE)
#                     last_check[pair] = current_candle_time
            
#         except Exception as e:
#             print(f"[{pair}] Error: {e}")
#             import traceback
#             # traceback.print_exc()
#             continue
    
#     print("\n" + "-" * 100)


# # ===================== MAIN EXECUTION =====================

# # Configuration
# ACCOUNT_BALANCE = 100  # dollars
# RISK_PER_TRADE = 5     # dollars ($5 risk per trade = 5% of $100)
# REWARD_PER_TRADE = 10  # dollars (1:2 risk/reward)

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "XAUUSD"]
# last_check = {}  # Track last candle time to avoid duplicate signals (global for scheduler)

# print("=" * 70)
# print("EMA CROSSOVER + RSI FILTER STRATEGY (RISK-BASED POSITION SIZING)")
# print("=" * 70)
# print(f"Account Balance: ${ACCOUNT_BALANCE}")
# print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
# print(f"Reward per Trade: ${REWARD_PER_TRADE} (1:{REWARD_PER_TRADE/RISK_PER_TRADE:.0f} R/R)")
# print(f"Strategy: 21/51 EMA on 5-Min Chart + RSI(14)")
# print(f"Entry: Bullish (RSI >= 60) | Bearish (RSI <= 40)")
# print(f"Stop Loss: Candle High/Low +/- 1.5× ATR(14)")
# print("=" * 70)

# connect_mt5(portable=True)  # Use portable=False if path issues

# # Create scheduler
# scheduler = BlockingScheduler()

# # Schedule the scan_markets function to run every minute
# # This will check for new 5-minute candles efficiently
# scheduler.add_job(
#     scan_markets,
#     trigger='cron',
#     minute='*',  # Run every minute
#     second='5',  # Start 5 seconds into each minute
#     id='market_scanner',
#     name='Scan markets for EMA+RSI signals',
#     max_instances=1  # Prevent overlapping executions
# )

# print("\n✓ Scheduler initialized")
# print("  Job: Scan markets every minute")
# print("  Press Ctrl+C to stop\n")

# try:
#     # Start the scheduler (blocking)
#     scheduler.start()

# except (KeyboardInterrupt, SystemExit):
#     print("\n\nStopping strategy...")
#     scheduler.shutdown()

# finally:
#     shutdown_mt5()


# """
# This below script is for 21 EMA crossover to the 51 EMA and also check for the RSI should be above or below (60, 40) range. And the stop loss is 1.5 ATR 
# with position sizing based on risk amount ($5 risk, $10 profit per trade).
# """


# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime, timezone
# import time
# import os
# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.triggers.cron import CronTrigger
# import os
# from dotenv import load_dotenv

# load_dotenv()  # loads .env file

# def connect_mt5(portable=True):
#     """Connect to MT5 with portable mode option"""
    
#     if portable:
#         mt5_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MetaTrader 5\terminal64.exe"
        
#         # Alternative paths - uncomment if needed:
#         # mt5_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
#         # mt5_path = os.getenv('MT5_PATH', mt5_path)
        
#         if not os.path.exists(mt5_path):
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
#             portable = False
    
#     if portable:
#         if not mt5.initialize(
#             path=mt5_path,
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#             timeout=60000,
#             portable=True
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
#     else:
#         if not mt5.initialize(
#             login=int(os.getenv("MT5_LOGIN")),
#             password=os.getenv("MT5_PASSWORD"),
#             server=os.getenv("MT5_SERVER"),
#         ):
#             error = mt5.last_error()
#             raise RuntimeError(f"MT5 initialize() failed: {error}")
    
#     account_info = mt5.account_info()
#     if account_info:
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance}")
#         print(f"  Leverage: 1:{account_info.leverage}")
#         pass
#     else:
#         print("✓ Connected to MT5")
#         pass


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


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
#         print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
#         pass
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
#         pass
    
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
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         print(f"⚠ Invalid ATR value: {current_atr}, cannot place trade")
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
#             print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
#             return
#     else:
#         if sl <= price:
#             print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
#             return
    
#     # Get contract size and point value
#     contract_size = symbol_info.trade_contract_size
#     point = symbol_info.point
    
#     # Calculate pip value per lot
#     # For forex pairs, 1 pip = 10 points (except JPY pairs where 1 pip = 100 points)
#     if "JPY" in symbol:
#         pip_multiplier = 10
#         pip_size = 0.01
#     else:
#         pip_multiplier = 10
#         pip_size = 0.0001
    
#     # SL distance in pips
#     sl_dist_pips = sl_dist_price / (point * pip_multiplier)
    
#     # Calculate pip value for 1 standard lot (100,000 units)
#     # For USD-based pairs or when account currency is USD
#     if symbol.startswith("USD"):
#         # USD is base - divide by price
#         pip_value_per_lot = (pip_size / price) * contract_size
#     elif symbol.endswith("USD"):
#         # USD is quote - multiply directly
#         pip_value_per_lot = pip_size * contract_size
#     else:
#         # Cross pairs
#         pip_value_per_lot = (pip_size / price) * contract_size
    
#     # Calculate required lot size for desired risk
#     # risk_amount = lot_size × sl_dist_pips × pip_value_per_lot
#     if sl_dist_pips > 0 and pip_value_per_lot > 0:
#         lot_size = risk_amount / (sl_dist_pips * pip_value_per_lot)
#     else:
#         print(f"⚠ Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
#         return
    
#     # Round to broker's volume step (usually 0.01)
#     volume_step = symbol_info.volume_step
#     lot_size = round(lot_size / volume_step) * volume_step
    
#     # Apply min/max lot size limits
#     if lot_size < symbol_info.volume_min:
#         print(f"⚠ Calculated lot size {lot_size:.2f} is below minimum {symbol_info.volume_min}")
#         lot_size = symbol_info.volume_min
#     elif lot_size > symbol_info.volume_max:
#         print(f"⚠ Calculated lot size {lot_size:.2f} exceeds maximum {symbol_info.volume_max}")
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
#             print("ERROR: Invalid take profit for bullish trade")
#             return
#     else:
#         if tp >= price:
#             print("ERROR: Invalid take profit for bearish trade")
#             return
    
#     # Calculate actual risk/reward in dollars
#     actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
#     actual_reward = lot_size * tp_dist_pips * pip_value_per_lot
    
#     print(f"\n{'='*70}")
#     print(f"TRADE SETUP - {symbol}")
#     print(f"{'='*70}")
#     print(f"Signal: {signal.upper()}")
#     print(f"Entry Price: {price:.5f}")
#     print(f"Stop Loss: {sl:.5f} (Distance: {sl_dist_pips:.1f} pips)")
#     print(f"Take Profit: {tp:.5f} (Distance: {tp_dist_pips:.1f} pips)")
#     print(f"")
#     print(f"Position Size: {lot_size:.2f} lots")
#     print(f"Contract Size: {contract_size:,.0f}")
#     print(f"Pip Value/Lot: ${pip_value_per_lot:.2f}")
#     print(f"")
#     print(f"Expected RISK: ${actual_risk:.2f} (Target: ${risk_amount:.2f})")
#     print(f"Expected REWARD: ${actual_reward:.2f} (Target: ${reward_amount:.2f})")
#     print(f"Risk/Reward Ratio: 1:{actual_reward/actual_risk:.2f}")
#     print(f"")
#     print(f"ATR(14): {current_atr:.5f} | ATR Buffer (1.5×): {atr_buffer:.5f}")
#     print(f"{'='*70}\n")

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
#             print(f"✗ Order send failed (returned None) with {filling_type}")
#             print(f"  MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  Order Ticket: {result.order}")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes.")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"\n[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  RSI: {curr_candle['rsi']:.2f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#     f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade with risk-based position sizing
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas, 
#                    risk_amount=risk_per_trade, reward_amount=reward_per_trade)


# def scan_markets():
#     """
#     Scan all currency pairs for trading signals.
#     This function is called by the scheduler.
#     """
#     current_time = datetime.now()
#     print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
    
#     for pair in currency_pair_list:
#         try:
#             if not is_market_open(pair):
#                 print(f"[{pair}] Market not open - skipping")
#                 continue
            
#             # Get current candle time
#             df = get_candle_data(pair, mt5.TIMEFRAME_M5, n=2)
#             if df is not None:
#                 current_candle_time = df.iloc[-1]['time']
                
#                 # Only check if new candle formed
#                 if pair not in last_check or last_check[pair] != current_candle_time:
#                     run_ema_strategy(pair, mt5.TIMEFRAME_M5, 
#                                    risk_per_trade=RISK_PER_TRADE, 
#                                    reward_per_trade=REWARD_PER_TRADE)
#                     last_check[pair] = current_candle_time
            
#         except Exception as e:
#             print(f"[{pair}] Error: {e}")
#             import traceback
#             # traceback.print_exc()
#             continue
    
#     print("\n" + "-" * 100)


# # ===================== MAIN EXECUTION =====================

# # Configuration
# ACCOUNT_BALANCE = 100  # dollars
# RISK_PER_TRADE = 5     # dollars ($5 risk per trade = 5% of $100)
# REWARD_PER_TRADE = 10  # dollars (1:2 risk/reward)

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "XAUUSD"]
# last_check = {}  # Track last candle time to avoid duplicate signals (global for scheduler)

# print("=" * 70)
# print("EMA CROSSOVER + RSI FILTER STRATEGY (RISK-BASED POSITION SIZING)")
# print("=" * 70)
# print(f"Account Balance: ${ACCOUNT_BALANCE}")
# print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
# print(f"Reward per Trade: ${REWARD_PER_TRADE} (1:{REWARD_PER_TRADE/RISK_PER_TRADE:.0f} R/R)")
# print(f"Strategy: 21/51 EMA on 5-Min Chart + RSI(14)")
# print(f"Entry: Bullish (RSI >= 60) | Bearish (RSI <= 40)")
# print(f"Stop Loss: Candle High/Low +/- 1.5× ATR(14)")
# print("=" * 70)

# connect_mt5(portable=True)  # Use portable=False if path issues

# # Create scheduler
# scheduler = BlockingScheduler()

# # Schedule the scan_markets function to run every minute
# # This will check for new 5-minute candles efficiently
# scheduler.add_job(
#     scan_markets,
#     trigger='cron',
#     minute='*',  # Run every minute
#     second='5',  # Start 5 seconds into each minute
#     id='market_scanner',
#     name='Scan markets for EMA+RSI signals',
#     max_instances=1  # Prevent overlapping executions
# )

# print("\n✓ Scheduler initialized")
# print("  Job: Scan markets every minute")
# print("  Press Ctrl+C to stop\n")

# try:
#     # Start the scheduler (blocking)
#     scheduler.start()

# except (KeyboardInterrupt, SystemExit):
#     print("\n\nStopping strategy...")
#     scheduler.shutdown()

# finally:
#     shutdown_mt5()


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
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
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
#         print("Connected to MT5")
#         print(f"Account: {account_info.login}")
#         print(f"Balance: ${account_info.balance}")
#         print(f"Leverage: 1:{account_info.leverage}")
#     else:
#         print("Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


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
#         print(f"Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         print(f"Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
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
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         print(f"Invalid ATR value: {current_atr}, cannot place trade")
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
#             print("ERROR: Invalid stop loss for bullish trade (SL >= Entry)")
#             return
#     else:
#         if sl <= price:
#             print("ERROR: Invalid stop loss for bearish trade (SL <= Entry)")
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
#         print(f"Invalid calculation: SL pips={sl_dist_pips}, pip_value={pip_value_per_lot}")
#         return
    
#     # Round to broker's volume step (usually 0.01)
#     volume_step = symbol_info.volume_step
#     lot_size = round(lot_size / volume_step) * volume_step
    
#     # Apply min/max lot size limits
#     if lot_size < symbol_info.volume_min:
#         print(f"Calculated lot size {lot_size:.2f} is below minimum {symbol_info.volume_min}")
#         lot_size = symbol_info.volume_min
#     elif lot_size > symbol_info.volume_max:
#         print(f"Calculated lot size {lot_size:.2f} exceeds maximum {symbol_info.volume_max}")
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
#             print("ERROR: Invalid take profit for bullish trade")
#             return
#     else:
#         if tp >= price:
#             print("ERROR: Invalid take profit for bearish trade")
#             return
    
#     # Calculate actual risk/reward in dollars
#     actual_risk = lot_size * sl_dist_pips * pip_value_per_lot
#     actual_reward = lot_size * tp_dist_pips * pip_value_per_lot
    
#     print(f"\n{'='*70}")
#     print(f"TRADE SETUP - {symbol}")
#     print(f"{'='*70}")
#     print(f"Signal: {signal.upper()}")
#     print(f"Entry Price: {price:.5f}")
#     print(f"Stop Loss: {sl:.5f} (Distance: {sl_dist_pips:.1f} pips)")
#     print(f"Take Profit: {tp:.5f} (Distance: {tp_dist_pips:.1f} pips)")
#     print(f"Position Size: {lot_size:.2f} lots")
#     print(f"Contract Size: {contract_size:,.0f}")
#     print(f"Pip Value/Lot: ${pip_value_per_lot:.2f}")
#     print(f"Expected RISK: ${actual_risk:.2f} (Target: ${risk_amount:.2f})")
#     print(f"Expected REWARD: ${actual_reward:.2f} (Target: ${reward_amount:.2f})")
#     print(f"Risk/Reward Ratio: 1:{actual_reward/actual_risk:.2f}")
#     print(f"ATR(14): {current_atr:.5f} | ATR Buffer (1.5×): {atr_buffer:.5f}")
#     print(f"{'='*70}\n")

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
#             print(f"Order send failed (returned None) with {filling_type}")
#             print(f"MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             print(f"{signal.upper()} trade placed successfully!")
#             print(f"  Order Ticket: {result.order}")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             print(f"Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print("Order failed with all filling modes.")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"\n[{symbol}]  {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  RSI: {curr_candle['rsi']:.2f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
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

# print("=" * 123)
# print("EMA CROSSOVER + RSI FILTER STRATEGY (RISK-BASED POSITION SIZING)")
# print("=" * 123)
# print(f"Account Balance: ${ACCOUNT_BALANCE}")
# print(f"Risk per Trade: ${RISK_PER_TRADE} ({RISK_PER_TRADE/ACCOUNT_BALANCE*100:.1f}% of balance)")
# print(f"Reward per Trade: ${REWARD_PER_TRADE} (1:{REWARD_PER_TRADE/RISK_PER_TRADE:.0f} R/R)")
# print("Strategy: 21/51 EMA on 5-Min Chart + RSI(14)")
# print("Entry: Bullish (RSI >= 60) | Bearish (RSI <= 40)")
# print("Stop Loss: Candle High/Low +/- 1.5× ATR(14)")
# print("=" * 123)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     print(f"[{pair}] Market not open - skipping")
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
#                 print(f"[{pair}] Error: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         # Wait for next check (check every 60 seconds)
#         print("\n" + "-" * 123)
#         time.sleep(60)

# except KeyboardInterrupt:
#     print("\n\nStopping strategy...")

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
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
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
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance}")
#     else:
#         print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


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
#         print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
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
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
#         return
    
#     # Entry price
#     price = tick.ask if signal == "bullish" else tick.bid
#     order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

#     # Calculate ATR (14 period)
#     df['atr'] = calculate_atr(df, period=14)
#     current_atr = curr_candle['atr'] if 'atr' in curr_candle else df['atr'].iloc[-1]
    
#     if pd.isna(current_atr) or current_atr <= 0:
#         print(f"⚠ Invalid ATR value: {current_atr}, cannot place trade")
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
    
#     print(f"Symbol: {symbol}, Point: {point}")
#     print(f"ATR(14): {current_atr:.5f}")
#     print(f"ATR Buffer (1.5×): {atr_buffer:.5f}")
#     print(f"Candle {'Low' if signal == 'bullish' else 'High'}: {curr_candle['low'] if signal == 'bullish' else curr_candle['high']:.5f}")
#     print(f"Calculated SL: {sl_price:.5f}")
#     print(f"Calculated SL dist: {sl_dist:.5f}")
#     # print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     # if sl_dist < min_distance:
#     #     print("⚠ SL too tight, using minimum distance")
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
#             print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

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
#             print(f"✗ Order send failed (returned None) with {filling_type}")
#             print(f"  MT5 Error: {error}")
#             continue
        
#         if result.retcode == mt5.TRADE_RETCODE_DONE:
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             print(f"  RSI: {curr_candle['rsi']:.2f}")
#             print(f"  ATR: {current_atr:.5f}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes.")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"[{symbol}] {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  RSI: {curr_candle['rsi']:.2f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# print("=" * 60)
# print("EMA CROSSOVER + RSI FILTER STRATEGY")
# print("21/51 EMA on 5-Min Chart + RSI(14)")
# print("Bullish: RSI >= 60 | Bearish: RSI <= 40")
# print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     print(f"[{pair}] Market not open - skipping")
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
#                 print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         print("\n" + "-" * 100)
#         time.sleep(60)

# except KeyboardInterrupt:
#     print("\n\nStopping strategy...")

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
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
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
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance}")
#     else:
#         print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


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
#         print(f"  ℹ Bullish crossover detected but RSI {curr['rsi']:.2f} < {rsi_threshold_bullish} (rejected)")
    
#     if prev['ema_21'] >= prev['ema_51'] and curr['ema_21'] < curr['ema_51']:
#         print(f"  ℹ Bearish crossover detected but RSI {curr['rsi']:.2f} > {rsi_threshold_bearish} (rejected)")
    
#     return "neutral", curr, df


# def place_ema_trade(symbol, signal, curr_candle, df, lot=0.1):
#     """
#     Place trade based on EMA crossover + RSI filter
#     SL: Current candle's high (bearish) or low (bullish)
#     TP: 2x SL distance
#     """
#     if signal not in ["bullish", "bearish"]:
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
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
    
#     print(f"Symbol: {symbol}, Point: {point}")
#     print(f"Calculated SL dist: {sl_dist:.5f} (from candle {'low' if signal == 'bullish' else 'high'})")
#     print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     if sl_dist < min_distance:
#         print("⚠ SL too tight, using minimum distance")
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
#             print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

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
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             print(f"  RSI: {curr_candle['rsi']:.2f}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes. Last: {result.retcode} - {result.comment}")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal (with RSI filter)
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No valid signal")
#         return
    
#     print(f"[{symbol}] {signal.upper()} CROSSOVER DETECTED (RSI CONFIRMED)!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  RSI: {curr_candle['rsi']:.2f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# print("=" * 60)
# print("EMA CROSSOVER + RSI FILTER STRATEGY")
# print("21/51 EMA on 5-Min Chart + RSI(14)")
# print("Bullish: RSI >= 60 | Bearish: RSI <= 40")
# print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     print(f"[{pair}] Market not open - skipping")
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
#                 print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         print("\n" + "-" * 60)
#         time.sleep(30)

# except KeyboardInterrupt:
#     print("\n\nStopping strategy...")

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
#             print(f"Warning: MT5 not found at {mt5_path}, trying default initialization")
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
#         print("✓ Connected to MT5")
#         print(f"  Account: {account_info.login}")
#         print(f"  Balance: ${account_info.balance}")
#     else:
#         print("✓ Connected to MT5")


# def shutdown_mt5():
#     mt5.shutdown()
#     print("MT5 disconnected")


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
#         print("No trade signal (neutral) — skipping trade.")
#         return

#     # Get symbol info
#     symbol_info = mt5.symbol_info(symbol)
#     if symbol_info is None:
#         print(f"Failed to get symbol info for {symbol}")
#         return
    
#     if not symbol_info.visible:
#         if not mt5.symbol_select(symbol, True):
#             print(f"Failed to add {symbol} to Market Watch")
#             return

#     tick = mt5.symbol_info_tick(symbol)
#     if tick is None:
#         print(f"Failed to get tick for {symbol}")
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
    
#     print(f"Symbol: {symbol}, Point: {point}")
#     print(f"Calculated SL dist: {sl_dist:.5f} (from candle {'low' if signal == 'bullish' else 'high'})")
#     print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
#     # Use the larger of calculated or minimum
#     if sl_dist < min_distance:
#         print("⚠ SL too tight, using minimum distance")
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
#             print("ERROR: Invalid stop levels for bullish trade")
#             return
#     else:
#         if sl <= price or tp >= price:
#             print("ERROR: Invalid stop levels for bearish trade")
#             return
    
#     print(f"Entry: {price} | SL: {sl} | TP: {tp}")
#     print(f"Risk/Reward: 1:{tp_dist/sl_dist:.1f}")

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
#             print(f"✓ {signal.upper()} trade placed successfully!")
#             print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#             print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#             return
#         elif result.retcode == 10018:
#             print(f"✗ Market is closed for {symbol}")
#             return
#         elif result.retcode != 10030:
#             print(f"✗ Order failed: {result.retcode} - {result.comment}")
#             return
    
#     print(f"✗ Order failed with all filling modes. Last: {result.retcode} - {result.comment}")


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
#         print(f"[{symbol}] Already have open position - skipping")
#         return
    
#     # Get data
#     df = get_candle_data(symbol, timeframe, n=100)
#     if df is None or len(df) < 52:
#         print(f"[{symbol}] Insufficient data")
#         return
    
#     # Detect signal
#     signal, curr_candle, df_with_emas = detect_ema_crossover(df)
    
#     if signal == "neutral":
#         print(f"[{symbol}] No crossover signal")
#         return
    
#     print(f"[{symbol}] 🎯 {signal.upper()} CROSSOVER DETECTED!")
#     print(f"  21 EMA: {curr_candle['ema_21']:.5f}")
#     print(f"  51 EMA: {curr_candle['ema_51']:.5f}")
#     print(f"  Candle: O={curr_candle['open']:.5f}, H={curr_candle['high']:.5f}, "
#           f"L={curr_candle['low']:.5f}, C={curr_candle['close']:.5f}")
    
#     # Place trade
#     place_ema_trade(symbol, signal, curr_candle, df_with_emas)


# # ===================== MAIN EXECUTION =====================

# currency_pair_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]

# print("=" * 60)
# print("EMA CROSSOVER STRATEGY (21/51 EMA on 5-Min Chart)")
# print("=" * 60)

# connect_mt5(portable=True)  # Use portable=False if path issues

# try:
#     last_check = {}  # Track last candle time to avoid duplicate signals
    
#     while True:
#         current_time = datetime.now()
#         print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for signals...")
        
#         for pair in currency_pair_list:
#             try:
#                 if not is_market_open(pair):
#                     print(f"[{pair}] Market not open - skipping")
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
#                 print(f"[{pair}] Error: {e}")
#                 continue
        
#         # Wait for next check (check every 30 seconds)
#         print("\n" + "-" * 60)
#         time.sleep(30)

# except KeyboardInterrupt:
#     print("\n\nStopping strategy...")

# finally:
#     shutdown_mt5()