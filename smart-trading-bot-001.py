import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

symbol = 'DOGEUSDT'
interval = '1min'
leverage = 50
initial_balance = 10.0
log_file = 'smart_log.txt'
model_path = 'dogeusdt_model.h5'

model = load_model(model_path)
scaler = StandardScaler()


# Logs
def log(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full = f"[{timestamp}] {message}"
    print(full)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(full + '\n')

# Fetch candle data

def get_candles(limit=100):
    try:
        url = 'https://itamir7.ir/proxy/cx.php/v1/market/kline'
        params = {
            'market': symbol,
            'type': interval,
            'limit': limit
        }
        r = requests.get(url, params=params)
        data = r.json()['data']
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', '_', '_'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        log(f"Error fetching data: {e}")
        return None

# Feature engineering

def add_features(df):
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(close=df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

# Strategy predictions

def strategy_predictions(df):
    latest = df.iloc[-1]
    pred = {'macd_ema': 2, 'rsi_bb': 2, 'ichimoku': 2}

    # 1. MACD + EMA
    if latest['macd'] > latest['macd_signal'] and latest['close'] > latest['ema_20']:
        pred['macd_ema'] = 1
    elif latest['macd'] < latest['macd_signal'] and latest['close'] < latest['ema_20']:
        pred['macd_ema'] = 0

    # 2. RSI + Bollinger
    if latest['rsi_14'] < 30 and latest['close'] < latest['bb_lower']:
        pred['rsi_bb'] = 1
    elif latest['rsi_14'] > 70 and latest['close'] > latest['bb_upper']:
        pred['rsi_bb'] = 0

    # 3. Ichimoku
    highs = df['high'][-52:]
    lows = df['low'][-52:]
    close = df['close'][-27]
    tenkan = (df['high'][-9:].max() + df['low'][-9:].min()) / 2
    kijun = (df['high'][-26:].max() + df['low'][-26:].min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (highs.max() + lows.min()) / 2

    if latest['close'] > senkou_a and latest['close'] > senkou_b and tenkan > kijun:
        pred['ichimoku'] = 1
    elif latest['close'] < senkou_a and latest['close'] < senkou_b and tenkan < kijun:
        pred['ichimoku'] = 0

    return pred

# AI model prediction

def ai_prediction(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'ema_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal']
    latest = df[features].values[-1].reshape(1, -1)
    scaled = scaler.fit_transform(df[features])  # فرض اینکه قبلا scaler ذخیره نشده
    last_scaled = scaled[-1].reshape(1, -1)
    prediction = model.predict(last_scaled, verbose=0)
    return np.argmax(prediction)

# Smart decision based on votes

def smart_decision(strategies, ai_decision):
    votes = list(strategies.values()) + [ai_decision]
    return max(set(votes), key=votes.count)

# Trading simulation

def simulate():
    balance = initial_balance
    in_position = False
    entry_price = 0
    direction = None

    log(f"🎯 شروع اجرای مجازی با سرمایه {balance} دلار...")

    while True:
        df = get_candles()
        if df is None or len(df) < 60:
            time.sleep(30)
            continue

        df = add_features(df)
        strategies = strategy_predictions(df)
        ai_dec = ai_prediction(df)
        decision = smart_decision(strategies, ai_dec)

        price = df['close'].iloc[-1]
        time_str = df.index[-1].strftime('%H:%M:%S')
        action_map = {0: 'Sell', 1: 'Buy', 2: 'Hold'}
        log(f"{time_str} | قیمت: {price:.5f} | پیش‌بینی: {action_map[decision]}")

        if not in_position:
            if decision in [0, 1]:
                direction = 'LONG' if decision == 1 else 'SHORT'
                entry_price = price
                in_position = True
                log(f"📥 ورود به پوزیشن {direction} در قیمت {price:.5f}")
        else:
            pnl = (price - entry_price) / entry_price * leverage if direction == 'LONG' else (entry_price - price) / entry_price * leverage
            dollar = pnl * balance
            log(f"📊 سود/زیان لحظه‌ای: {pnl*100:.2f}% | ${dollar:.2f}")

            # Exit logic
            if pnl > 0.2 or pnl < -0.1:
                in_position = False
                balance += dollar
                log(f"📤 خروج از پوزیشن {direction} | قیمت خروج: {price:.5f} | موجودی جدید: ${balance:.2f}")

        time.sleep(60)

simulate()
