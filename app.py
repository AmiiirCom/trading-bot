import requests
import time
from datetime import datetime

# Settings
symbol = 'PEPEUSDT'
interval = '1min'
leverage = 50
initial_balance = 10.0
log_file = 'trading_log.txt'

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(full_message + '\n')

def get_candles(limit=120):
    try:
        url = 'https://api.coinex.com/v1/market/kline'
        params = {
            'market': symbol,
            'type': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['code'] != 0:
            log(f"API error: {data}")
            return None

        candles = []
        for item in data['data']:
            candles.append({
                'timestamp': item[0] * 1000,
                'open': float(item[1]),
                'close': float(item[2]),
                'high': float(item[3]),
                'low': float(item[4]),
                'volume': float(item[5])
            })

        return candles
    except Exception as e:
        log(f"Error fetching candles: {e}")
        return None

def ichimoku(candles):
    if len(candles) < 52:
        return None

    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    closes = [c['close'] for c in candles]

    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
    chikou = closes[-27] if len(closes) > 26 else None

    return {
        'tenkan_sen': tenkan,
        'kijun_sen': kijun,
        'senkou_span_a': senkou_a,
        'senkou_span_b': senkou_b,
        'chikou_span': chikou
    }

def calculate_profit(current_price, entry_price, leverage, balance, is_long):
    change = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
    return change * leverage, change * leverage * balance

def simulate_live_trading():
    balance = initial_balance
    in_position = False
    entry_price = 0
    entry_time = None
    direction = None

    log("Starting Ichimoku strategy trading bot...")

    while True:
        candles = get_candles(120)
        if not candles or len(candles) < 52:
            log("Waiting for enough data...")
            time.sleep(30)
            continue

        latest = candles[-1]
        price = latest['close']
        time_str = datetime.fromtimestamp(latest['timestamp'] / 1000).strftime('%H:%M:%S')
        log(f"{time_str} | Open: {latest['open']} | Close: {price} | Volume: {latest['volume']}")

        ichimoku_values = ichimoku(candles)
        if ichimoku_values is None:
            log("Not enough data for Ichimoku calculation.")
            time.sleep(30)
            continue

        tenkan = ichimoku_values['tenkan_sen']
        kijun = ichimoku_values['kijun_sen']
        senkou_a = ichimoku_values['senkou_span_a']
        senkou_b = ichimoku_values['senkou_span_b']

        if not in_position:
            if price > senkou_a and price > senkou_b and tenkan > kijun:
                direction = 'LONG'
                entry_price = price
                entry_time = datetime.now()
                in_position = True
                log(f"Entered LONG position at {entry_price}")
            elif price < senkou_a and price < senkou_b and tenkan < kijun:
                direction = 'SHORT'
                entry_price = price
                entry_time = datetime.now()
                in_position = True
                log(f"Entered SHORT position at {entry_price}")
        else:
            is_long = (direction == 'LONG')
            leveraged_percent, dollar_profit = calculate_profit(price, entry_price, leverage, balance, is_long)
            log(f"Unrealized PnL: {leveraged_percent*100:.2f}% | ${dollar_profit:.2f}")

            if leveraged_percent >= 0.15:
                log(f"Closing {direction} position at {price} due to 15% profit.")
                balance += dollar_profit
                log(f"Profit: ${dollar_profit:.2f} | New balance: ${balance:.2f}")
                in_position = False
            elif leveraged_percent <= -0.10:
                log(f"Closing {direction} position at {price} due to -10% stop-loss.")
                balance += dollar_profit
                log(f"Loss: ${dollar_profit:.2f} | New balance: ${balance:.2f}")
                in_position = False

        time.sleep(30)

simulate_live_trading()
