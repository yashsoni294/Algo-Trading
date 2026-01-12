import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import schedule
import time
from datetime import datetime, timezone


def connect_mt5():
    if not mt5.initialize(login = 5044120156, password = "ZbMm@6Ib", server = "MetaQuotes-Demo"):
        raise RuntimeError(" !!! Error >>> MT5 initialize() failed")
    print("Good! Connected to MT5")


def shutdown_mt5():
    mt5.shutdown()
    print("MT5 disconnected")


def get_fifteen_min_data(symbol="EURUSD", n=10):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def detect_engulfing(df):
    if len(df) < 2:
        return "neutral"
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev['close'] < prev['open'] and curr['close'] > curr['open']:
        if curr['close'] > prev['open'] and curr['open'] < prev['close']:
            return "bullish"
    if prev['close'] > prev['open'] and curr['close'] < curr['open']:
        if curr['open'] > prev['close'] and curr['close'] < prev['open']:
            return "bearish"
    # return "neutral"
    return "bearish"


def place_trade(symbol, signal, df, lot=0.1):
    if signal not in ["bullish", "bearish"]:
        print("No trade signal (neutral) — skipping trade.")
        return

    # Check if market is open
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return
    
    if not symbol_info.visible:
        print(f"Symbol {symbol} is not visible in Market Watch")
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to add {symbol} to Market Watch")
            return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for {symbol}")
        return
    
    price = tick.ask if signal == "bullish" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "bullish" else mt5.ORDER_TYPE_SELL

    curr = df.iloc[-1]
    sl_price = curr['low'] if signal == "bullish" else curr['high']
    sl_dist = abs(price - sl_price)
    
    # Set MINIMUM stop distances based on typical broker requirements
    point = symbol_info.point
    
    # Minimum 15 pips for major pairs, 30 for exotics
    if symbol in ['USDCNH', 'USDZAR', 'USDTRY']:
        min_pips = 30
    else:
        min_pips = 15
    
    min_distance = min_pips * point * 10  # Convert pips to price
    
    print(f"Symbol: {symbol}, Point: {point}")
    print(f"Calculated SL dist: {sl_dist:.5f}")
    print(f"Minimum required: {min_distance:.5f} ({min_pips} pips)")
    
    # Use the larger of calculated or minimum
    if sl_dist < min_distance:
        print("SL too tight, using minimum distance")
        sl_dist = min_distance
    
    tp_dist = 2 * sl_dist
    
    # Calculate final SL/TP
    if signal == "bullish":
        sl = round(price - sl_dist, symbol_info.digits)
        tp = round(price + tp_dist, symbol_info.digits)
    else:
        sl = round(price + sl_dist, symbol_info.digits)
        tp = round(price - tp_dist, symbol_info.digits)
    
    # Validate stops are on correct side
    if signal == "bullish":
        if sl >= price or tp <= price:
            print("ERROR: Invalid stop levels for bullish trade")
            return
    else:
        if sl <= price or tp >= price:
            print("ERROR: Invalid stop levels for bearish trade")
            return
    
    print(f"Final: Price={price}, SL={sl}, TP={tp}")
    print(f"Distances: SL={abs(price-sl):.5f}, TP={abs(price-tp):.5f}")

    # For market orders, try filling modes in order of compatibility
    filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    
    for filling_type in filling_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 234000,
            "comment": f"{signal} engulfing",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✓ {signal.capitalize()} trade placed successfully!")
            return
        elif result.retcode == 10018:
            print(f"✗ Market is closed for {symbol}")
            return
        elif result.retcode != 10030:  # If not filling mode error, don't retry
            print(f"✗ Order failed: {result.retcode} - {result.comment}")
            return
    
    print(f"✗ Order failed with all filling modes. Last: {result.retcode} - {result.comment}")


def is_market_open(symbol: str) -> bool:
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[{symbol}] Symbol info not found")
        return False

    if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        print(f"[{symbol}] Trade mode not FULL")
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[{symbol}] Cannot select symbol")
            return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[{symbol}] No tick data")
        return False

    if tick.bid <= 0 or tick.ask <= 0:
        print(f"[{symbol}] Invalid bid/ask (bid={tick.bid}, ask={tick.ask})")
        return False

    # Tick freshness
    tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now = datetime.now(timezone.utc)

    if (now - tick_time).total_seconds() > 30:
        print(f"[{symbol}] Stale tick ({(now - tick_time).total_seconds():.1f}s old)")
        return False

    # Rollover protection using broker time
    if tick_time.hour == 23 and tick_time.minute >= 55:
        print(f"[{symbol}] Rollover window — skipping")
        return False

    if tick_time.hour == 0 and tick_time.minute <= 5:
        print(f"[{symbol}] Rollover window — skipping")
        return False

    return True


def run_strategy(symbol):
    df = get_fifteen_min_data(symbol, n=3)
    # print(df.tail(2))
    print(df)

    signal = detect_engulfing(df)
    print(f"Signal: {signal}")

    place_trade(symbol, signal, df)


# schedule.every(10).seconds.do(run_strategy)


print("------> Strategy scheduler is running every 10 seconds...")


currency_pair_list = ["USDJPY", "EURUSD", "GBPUSD", "USDCNH", "USDCHF", "AUDUSD"]


connect_mt5()

try:
    while True:
        for pair in currency_pair_list:
            print(f"\nValidating for {pair}...")

            if not is_market_open(pair):
                print(f"[{pair}] Market not tradable now — skipping")
                continue

            run_strategy(symbol=pair)

        time.sleep(10)

except KeyboardInterrupt:
    print("Stopping strategy...")

finally:
    shutdown_mt5()