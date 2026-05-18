from scipy.stats import pearsonr
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, Bidirectional, Attention, GlobalAveragePooling1D,
                                     MultiHeadAttention, LayerNormalization, Add, Flatten)
from tensorflow.keras.models import Sequential, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (roc_curve, roc_auc_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
START_DATE = '2023-05-08'   # 3 years of data
BACKTEST_D = 30             # Last N days for backtest
SEQ_LEN = 30                # Lookback window
EPOCHS = 60
BATCH = 32
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15

# Dynamic TP / SL Config
SL_ATR_MULT = 1.5          # SL distance: 1.5x ATR to avoid normal fluctuations
RR_RATIO = 2.0             # 2:1 Reward to Risk -> TP distance: 3.0x ATR

# ══════════════════════════════════════════════════════════════════════
#  1. FETCH DATA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print(
    f"  BTC-USD LSTM SIGNAL ENGINE  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═"*72)
print("  Fetching BTC-USD, GLD data (with synthetic fallback)...")


def fetch_or_synth(ticker, start, n_synth=1200, seed=42, base=40000, vol=0.035):
    try:
        raw = yf.download(ticker, start=start, interval='1d', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if len(raw) > 100:
            print(f"    ✓ {ticker}: {len(raw)} real rows")
            return raw
    except Exception:
        pass
    print(f"    ⚠ {ticker}: network blocked — using synthetic BALANCED GBM data")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_synth)

    vol_t = np.ones(n_synth) * vol
    for i in range(1, n_synth):
        shock = rng.normal(0, vol_t[i-1])
        vol_t[i] = np.clip(0.94 * vol_t[i-1] + 0.06 * abs(shock), 0.008, 0.12)

    log_rets = rng.normal(0.0, vol_t, n_synth)

    for i in range(0, n_synth, 120):
        end = min(i + 120, n_synth)
        sign = 1 if (i // 120) % 2 == 0 else -1
        log_rets[i:end] += sign * 0.0005

    close = base * np.exp(np.cumsum(log_rets))
    high = close * np.exp(np.abs(rng.normal(0, vol_t/2, n_synth)))
    low = close * np.exp(-np.abs(rng.normal(0, vol_t/2, n_synth)))
    open_ = close * np.exp(rng.normal(0, vol_t/3, n_synth))
    vol_v = rng.lognormal(20, 1, n_synth)
    df_s = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': vol_v},
                        index=pd.DatetimeIndex(dates))
    up_pct = (np.diff(close) > 0).mean() * 100
    print(f"    ↑ Synthetic up-days: {up_pct:.1f}%  (target ~50%)")
    return df_s


btc = fetch_or_synth('BTC-USD', START_DATE, seed=42,  base=42000, vol=0.038)
gld = fetch_or_synth('GLD',     START_DATE, seed=123, base=170,   vol=0.010)

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)
if isinstance(gld.columns, pd.MultiIndex):
    gld.columns = gld.columns.get_level_values(0)

df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
gld_close = gld['Close'].reindex(df.index, method='ffill')

print(
    f"  Rows loaded: {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

# ══════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
c, h, lo, o, v = df.Close, df.High, df.Low, df.Open, df.Volume

# Add ATR for Dynamic TP/SL and Volatility measurement
tr = np.maximum(h - lo, np.maximum(abs(h - c.shift(1)), abs(lo - c.shift(1))))
df['atr'] = tr.rolling(14).mean()
df['atr_pct'] = df['atr'] / c

df['ret1'] = c.pct_change()
df['ret2'] = c.pct_change(2)
df['ret5'] = c.pct_change(5)
df['ret10'] = c.pct_change(10)
df['ret21'] = c.pct_change(21)

# Intermediate Volatility
rv5_calc = df['ret1'].rolling(5).std()
rv21_calc = df['ret1'].rolling(21).std()

# Kept Parkinson Volatility
df['parkinson'] = (np.log(h/lo)**2 / (4*np.log(2))).rolling(5).mean()

df['vol_ratio'] = v / v.rolling(20).mean()
df['vol_spike'] = (v / v.rolling(5).mean()).clip(0, 5)
df['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
df['vol_price_div'] = df['vol_ratio'] / \
    (rv5_calc.replace(0, np.nan) * 100 + 0.001)

df['gap'] = (o - c.shift(1)) / c.shift(1)
df['body'] = (c - o).abs() / (h - lo + 1e-9)
df['upper_wick'] = (h - c.clip(upper=h)) / (h - lo + 1e-9)
df['lower_wick'] = (c.clip(lower=lo) - lo) / (h - lo + 1e-9)
df['hl_range'] = (h - lo) / c

df['sma5'] = c.rolling(5).mean()
df['sma20'] = c.rolling(20).mean()
df['sma50'] = c.rolling(50).mean()

# Rolling Z-Scores
df['z_sma5'] = (c - df['sma5']) / (c.rolling(5).std() + 1e-9)
df['z_sma20'] = (c - df['sma20']) / (c.rolling(20).std() + 1e-9)
df['z_sma50'] = (c - df['sma50']) / (c.rolling(50).std() + 1e-9)

df['sma_cross'] = (df['sma5'] - df['sma20']) / df['sma20']
df['dist_52h'] = c / h.rolling(252).max() - 1
df['dist_52l'] = c / lo.rolling(252).min() - 1

btc_gold_ratio = c / gld_close.clip(lower=0.01)
df['btc_gold_ret'] = btc_gold_ratio.pct_change()
df['btc_gold_trend'] = btc_gold_ratio / btc_gold_ratio.rolling(20).mean() - 1

df['mom_accel'] = df['ret5'] - df['ret10']
df['vol_regime'] = rv5_calc / (rv21_calc + 1e-9)
df['mean_rev'] = -df['ret5'] * (1 / (rv5_calc + 0.001))

for lag in [1, 2, 3, 5]:
    df[f'ret1_lag{lag}'] = df['ret1'].shift(lag)

# Added Day-of-Week Sine/Cosine Encoding
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

df['target'] = (df['ret1'].shift(-1) > 0).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

FEATS = [
    'ret1', 'ret2', 'ret5', 'ret10', 'ret21',
    'parkinson',
    'vol_ratio', 'vol_spike', 'vol_trend', 'vol_price_div',
    'gap', 'body', 'upper_wick', 'lower_wick', 'hl_range',
    'z_sma5', 'z_sma20', 'z_sma50', 'sma_cross',
    'dist_52h', 'dist_52l',
    'btc_gold_ret', 'btc_gold_trend',
    'mom_accel', 'vol_regime', 'mean_rev',
    'ret1_lag1', 'ret1_lag2', 'ret1_lag3', 'ret1_lag5',
    'atr_pct',
    'day_sin', 'day_cos'
]
print(f"  Features engineered: {len(FEATS)}")

# ══════════════════════════════════════════════════════════════════════
#  3. SPLIT & SEQUENCE
# ══════════════════════════════════════════════════════════════════════
X_all = df[FEATS].values
y_all = df['target'].values

X_main = X_all[:-BACKTEST_D]
y_main = y_all[:-BACKTEST_D]
n = len(X_main)

n_train = int(n * TRAIN_SPLIT)
n_val = int(n * (TRAIN_SPLIT + VAL_SPLIT))

scaler = StandardScaler().fit(X_main[:n_train])
Xs_all = scaler.transform(X_all)
Xs_main = scaler.transform(X_main)


def make_seqs(X, y, sl):
    xs, ys = [], []
    for i in range(len(X) - sl):
        xs.append(X[i:i+sl])
        ys.append(y[i+sl])
    return np.array(xs), np.array(ys)


Xseq, yseq = make_seqs(Xs_main, y_main, SEQ_LEN)
Xtr, ytr = Xseq[:n_train-SEQ_LEN], yseq[:n_train-SEQ_LEN]
Xval, yval = Xseq[n_train-SEQ_LEN:n_val -
                  SEQ_LEN], yseq[n_train-SEQ_LEN:n_val-SEQ_LEN]
Xte, yte = Xseq[n_val-SEQ_LEN:], yseq[n_val-SEQ_LEN:]

print(
    f"  Train: {len(Xtr)} | Val: {len(Xval)} | Test: {len(Xte)} | Backtest: {BACKTEST_D} days")

# ══════════════════════════════════════════════════════════════════════
#  4. MODEL: Lean Bidirectional LSTM ("Thought Vector")
# ══════════════════════════════════════════════════════════════════════


def build_model(input_shape):
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)


cw = dict(enumerate(compute_class_weight(
    'balanced', classes=np.unique(ytr), y=ytr)))
model = build_model((SEQ_LEN, len(FEATS)))

model.compile(optimizer=Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

print(f"\n  Model parameters: {model.count_params():,}")

callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True,
                  monitor='val_auc', mode='max'),
    ReduceLROnPlateau(patience=6, factor=0.5, min_lr=1e-5,
                      monitor='val_auc', mode='max')
]

print("  Training base model...\n")
history = model.fit(
    Xtr, ytr,
    epochs=EPOCHS, batch_size=BATCH,
    validation_data=(Xval, yval),
    class_weight=cw,
    callbacks=callbacks,
    verbose=1
)

# ══════════════════════════════════════════════════════════════════════
#  5 & 6. CALIBRATION + BASE THRESHOLD
# ══════════════════════════════════════════════════════════════════════
print("\n  Evaluating raw probabilities...")


def calibrate(raw):
    return np.clip(np.asarray(raw, dtype=float), 0.01, 0.99)


raw_tr_probs = model.predict(Xtr,  verbose=0).flatten()
raw_val_probs = model.predict(Xval, verbose=0).flatten()
raw_te_probs = model.predict(Xte,  verbose=0).flatten()

cal_tr, cal_val, cal_te = calibrate(raw_tr_probs), calibrate(
    raw_val_probs), calibrate(raw_te_probs)

fpr, tpr, thrs = roc_curve(yte, cal_te)
auc_te = roc_auc_score(yte, cal_te)

val_median = float(np.median(cal_val))
youden_thr = float(thrs[np.argmax(tpr - fpr)])
THR = float(np.clip(0.5 * val_median + 0.5 * youden_thr, 0.35, 0.65))

print(f"  Base Model Center (THR): {THR:.3f}")

# ══════════════════════════════════════════════════════════════════════
#  7. METRICS — TRAIN / VAL / TEST
# ══════════════════════════════════════════════════════════════════════


def get_metrics(X, y, label):
    raw = model.predict(X, verbose=0).flatten()
    cal_p = calibrate(raw)
    pred = (cal_p >= THR).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    auc = roc_auc_score(y, cal_p)
    cm = confusion_matrix(y, pred)
    return dict(label=label, acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                probs=cal_p, preds=pred, cm=cm, raw=raw)


m_tr = get_metrics(Xtr, ytr, "TRAIN")
m_val = get_metrics(Xval, yval, "VAL  ")
m_te = get_metrics(Xte, yte, "TEST ")

# ══════════════════════════════════════════════════════════════════════
#  8. BACKTEST — Forward Validation, Dynamic TP/SL & Trend Adjustments
# ══════════════════════════════════════════════════════════════════════
print("\n  Running Forward Validation Backtest (Refining model & Dynamic Exits)...")

bdts = df.index[-BACKTEST_D:]
bcls = df.Close.values[-BACKTEST_D:]
bopen = df.Open.values[-BACKTEST_D:]
bhigh = df.High.values[-BACKTEST_D:]
blow = df.Low.values[-BACKTEST_D:]
batr = df.atr.values[-BACKTEST_D:]

sizes = np.zeros(BACKTEST_D)
bp = np.zeros(BACKTEST_D)
bp_raw = np.zeros(BACKTEST_D)
sr_daily = np.zeros(BACKTEST_D)
sl_arr = np.zeros(BACKTEST_D)
tp_arr = np.zeros(BACKTEST_D)
bh_ret = df['ret1'].values[-BACKTEST_D:]
labels_bt = ["CASH  "] * BACKTEST_D

current_center = THR

for i in range(BACKTEST_D):
    idx_in_all = len(df) - BACKTEST_D + i

    # ── A. FORWARD VALIDATION & SMOOTHED ROLLING THRESHOLD
    if i > 0 and i % 5 == 0:
        roll_X_raw = Xs_all[idx_in_all-252-SEQ_LEN: idx_in_all-1]
        roll_y_raw = y_all[idx_in_all-252-SEQ_LEN: idx_in_all-1]
        roll_X, roll_y = make_seqs(roll_X_raw, roll_y_raw, SEQ_LEN)

        model.fit(roll_X, roll_y, epochs=1, batch_size=BATCH, verbose=0)

        recent_X_raw = Xs_all[idx_in_all-30-SEQ_LEN: idx_in_all-1]
        recent_y_raw = y_all[idx_in_all-30-SEQ_LEN: idx_in_all-1]
        rX, _ = make_seqs(recent_X_raw, recent_y_raw, SEQ_LEN)
        if len(rX) > 0:
            recent_preds = calibrate(model.predict(rX, verbose=0).flatten())
            new_median = float(np.median(recent_preds))
            current_center = (current_center * 0.7) + (new_median * 0.3)

    today_seq = Xs_all[idx_in_all-SEQ_LEN: idx_in_all][np.newaxis, :, :]
    raw_pred = float(model.predict(today_seq, verbose=0).flatten()[0])
    prob = float(calibrate(np.array([raw_pred]))[0])

    bp_raw[i] = raw_pred
    bp[i] = prob

    recent_atr_pct = df['atr_pct'].iloc[idx_in_all-1]
    vol_buffer = np.clip(recent_atr_pct * 0.5, 0.005, 0.02)

    dyn_conf_high = current_center + vol_buffer
    dyn_conf_low = current_center - vol_buffer

    trend_state = df['sma5'].iloc[idx_in_all-1] / \
        df['sma50'].iloc[idx_in_all-1] - 1

    if trend_state > 0.02:
        dyn_conf_low -= 0.025
    elif trend_state < -0.02:
        dyn_conf_high += 0.025

    if prob > dyn_conf_high:
        sizes[i] = 1.0
        labels_bt[i] = "LONG  "
    elif prob < dyn_conf_low:
        sizes[i] = -1.0
        labels_bt[i] = "SHORT "
    else:
        sizes[i] = 0.0
        labels_bt[i] = "CASH  "

    # ── C. DYNAMIC TP/SL EXECUTION (Risk 1 : Reward 2)
    entry_px = bopen[i]
    today_high = bhigh[i]
    today_low = blow[i]
    today_close = bcls[i]
    today_atr = batr[i]

    trade_ret = 0.0
    if sizes[i] == 1.0:  # LONG
        sl_px = entry_px - (today_atr * SL_ATR_MULT)
        tp_px = entry_px + (today_atr * SL_ATR_MULT * RR_RATIO)
        sl_arr[i] = sl_px
        tp_arr[i] = tp_px
        if today_low <= sl_px:
            trade_ret = (sl_px - entry_px) / entry_px
        elif today_high >= tp_px:
            trade_ret = (tp_px - entry_px) / entry_px
        else:
            trade_ret = (today_close - entry_px) / entry_px

    elif sizes[i] == -1.0:  # SHORT
        sl_px = entry_px + (today_atr * SL_ATR_MULT)
        tp_px = entry_px - (today_atr * SL_ATR_MULT * RR_RATIO)
        sl_arr[i] = sl_px
        tp_arr[i] = tp_px
        if today_high >= sl_px:
            trade_ret = (entry_px - sl_px) / entry_px
        elif today_low <= tp_px:
            trade_ret = (entry_px - tp_px) / entry_px
        else:
            trade_ret = (entry_px - today_close) / entry_px
    else:
        sl_arr[i] = np.nan
        tp_arr[i] = np.nan

    sr_daily[i] = trade_ret

cum_strat = (1 + sr_daily).cumprod() - 1
cum_bh = (1 + bh_ret).cumprod() - 1
labels = labels_bt


def sharpe(rets, ann=365):
    rets = np.array(rets)
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(ann))


sh_strat = sharpe(sr_daily)
sh_bh = sharpe(bh_ret)

# ══════════════════════════════════════════════════════════════════════
#  9. TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════
GRN = "\033[92m"
RED = "\033[91m"
BLU = "\033[94m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"
RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"

print("\n" + "═"*98)
print(f"{BLD}  MODEL PERFORMANCE REPORT{RST}")
print("═"*98)
print(f"\n  {'METRIC':<18} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}")

for key, label in [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'),
                   ('f1', 'F1-Score'), ('auc', 'ROC-AUC')]:
    tr_v = m_tr[key]*100
    val_v = m_val[key]*100
    te_v = m_te[key]*100
    te_col = GRN if te_v >= 55 else (YEL if te_v >= 50 else RED)
    print(f"  {label:<18} {tr_v:>9.1f}% {val_v:>9.1f}% {te_col}{te_v:>9.1f}%{RST}")

print(f"\n  Confusion Matrix (TEST):")
cm = m_te['cm']
print(f"  {'':10} Pred:0   Pred:1")
print(f"  True:0    {cm[0, 0]:>5}    {cm[0, 1]:>5}")
print(f"  True:1    {cm[1, 0]:>5}    {cm[1, 1]:>5}")

p_test = m_te['probs']
print(f"\n  Confidence Distribution (TEST set, {len(p_test)} samples):")
bins = [(0, .4, '<40% (Strong Short)'), (0.4, .46, '40–46% (Weak Short)'),
        (0.46, .54, '46–54% (Neutral)'), (0.54, .60, '54–60% (Weak Long)'), (0.60, 1, '60%+ (Strong Long)')]
for lo_b, hi_b, label in bins:
    cnt = ((p_test >= lo_b) & (p_test < hi_b)).sum()
    bar = "█"*int(cnt/len(p_test)*40)
    print(f"  {label:<28} {cnt:>4}  {bar}")

print("\n" + "═"*98)
print(f"{BLD}  30-DAY BACKTEST  |  Trend-Adjusted Volatility Zone{RST}")
print("═"*98)

# Expanded header to fit SL and TP
hdr = (f"{'DATE':<12} {'PRICE':>9} {'CONF':>6} {'RAW':>6} "
       f"{'SIGNAL':<7} {'SIZE':>4} {'SL ($)':>8} {'TP ($)':>8} {'DAILY':>7} {'CUMUL':>8}  ST")
print(f"\n  {hdr}")
print("  " + "─"*95)

for i in range(BACKTEST_D):
    sig = labels[i]
    sz = sizes[i]
    conf = bp[i]
    raw = bp_raw[i]
    daily = sr_daily[i]*100
    cum = cum_strat[i]*100

    sl_str = f"{sl_arr[i]:>8.0f}" if not np.isnan(sl_arr[i]) else "       -"
    tp_str = f"{tp_arr[i]:>8.0f}" if not np.isnan(tp_arr[i]) else "       -"

    if sz > 0:
        col = GRN
    elif sz < 0:
        col = RED
    else:
        col = BLU

    if sz == 0:
        status = f"{DIM}–{RST}"
    elif daily > 0:
        status = f"{GRN}✓{RST}"
    else:
        status = f"{RED}✗{RST}"

    daily_col = GRN if daily > 0 else (RED if daily < 0 else DIM)
    cum_col = GRN if cum > 0 else RED

    print(f"  {str(bdts[i].date()):<12} ${bcls[i]:>8,.0f} "
          f"{conf*100:>5.1f}% {raw*100:>5.1f}% "
          f"{col}{sig:<7}{RST} {sz:>+4.1f} "
          f"{DIM}{sl_str}{RST} {DIM}{tp_str}{RST} "
          f"{daily_col}{daily:>6.2f}%{RST} "
          f"{cum_col}{cum:>7.2f}%{RST}  {status}")

print("  " + "─"*95)
wins = sum((sr_daily[i] > 0) for i in range(BACKTEST_D) if sizes[i] != 0)
active = sum(abs(sizes[i]) > 0 for i in range(BACKTEST_D))

print(
    f"\n  {'STRATEGY RETURN':<30} {GRN if cum_strat[-1] > 0 else RED}{cum_strat[-1]*100:>+8.2f}%{RST}")
print(
    f"  {'BUY & HOLD':<30} {GRN if cum_bh[-1] > 0 else RED}{cum_bh[-1]*100:>+8.2f}%{RST}")
print(
    f"  {'ALPHA':<30} {GRN if cum_strat[-1] > cum_bh[-1] else RED}{(cum_strat[-1]-cum_bh[-1])*100:>+8.2f}%{RST}")
print(f"  {'SHARPE (Strategy, ann.)':<30} {GRN if sh_strat > 0 else RED}{sh_strat:>+8.2f}{RST}")
print(f"  {'SHARPE (Buy & Hold, ann.)':<30} {GRN if sh_bh > 0 else RED}{sh_bh:>+8.2f}{RST}")
print(f"  {'WIN RATE (active signals)':<30} {wins}/{max(active, 1)} = {wins/max(active, 1)*100:.1f}%")
print(f"  {'ACTIVE DAYS':<30} {active}/{BACKTEST_D}")

# Tomorrow
tm_raw = float(model.predict(
    Xs_all[-SEQ_LEN:][np.newaxis, :, :], verbose=0).flatten()[0])
tm_conf = float(calibrate(np.array([tm_raw]))[0])

tm_atr_pct = df['atr_pct'].iloc[-1]
tm_vol_buffer = np.clip(tm_atr_pct * 0.5, 0.005, 0.02)
tm_high = current_center + tm_vol_buffer
tm_low = current_center - tm_vol_buffer

tm_trend_state = df['sma5'].iloc[-1] / df['sma50'].iloc[-1] - 1
if tm_trend_state > 0.02:
    tm_low -= 0.025
elif tm_trend_state < -0.02:
    tm_high += 0.025

if tm_conf > tm_high:
    tm_sz, tm_sig = 1.0, "LONG  "
elif tm_conf < tm_low:
    tm_sz, tm_sig = -1.0, "SHORT "
else:
    tm_sz, tm_sig = 0.0, "CASH  "

# Calculate Tomorrow's Targets based on the current close
tm_price = df['Close'].iloc[-1]
tm_atr = df['atr'].iloc[-1]

if tm_sz == 1.0:
    tm_sl = tm_price - (tm_atr * SL_ATR_MULT)
    tm_tp = tm_price + (tm_atr * SL_ATR_MULT * RR_RATIO)
    tm_exit_str = f"  |  SL: ${tm_sl:,.0f}  |  TP: ${tm_tp:,.0f}"
elif tm_sz == -1.0:
    tm_sl = tm_price + (tm_atr * SL_ATR_MULT)
    tm_tp = tm_price - (tm_atr * SL_ATR_MULT * RR_RATIO)
    tm_exit_str = f"  |  SL: ${tm_sl:,.0f}  |  TP: ${tm_tp:,.0f}"
else:
    tm_exit_str = ""

col_tm = GRN if tm_sz > 0 else (RED if tm_sz < 0 else BLU)
print("\n" + "═"*98)
print(f"  {BLD}TOMORROW'S SIGNAL:{RST}  {col_tm}{BLD}{tm_sig.strip()}{RST}  "
      f"| Confidence: {tm_conf*100:.1f}%  | Raw: {tm_raw*100:.1f}%  | Size: {tm_sz:+.1f}x{tm_exit_str}")
print(
    f"  (Dyn Long Thr: {tm_high*100:.1f}% | Dyn Short Thr: {tm_low*100:.1f}%)")
print("═"*98)

# ══════════════════════════════════════════════════════════════════════
#  10. CHARTS
# ══════════════════════════════════════════════════════════════════════
print("\n  Generating charts...")

fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK = '#0d1117'
PANEL = '#161b22'
GRID = '#30363d'
TEXT = '#e6edf3'
DIM_T = '#8b949e'
GRNC = '#3fb950'
REDC = '#f85149'
BLUC = '#58a6ff'
YELC = '#d29922'
ORGC = '#fb8500'


def ax_style(ax, title):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=DIM_T, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=8)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
    ax.xaxis.label.set_color(DIM_T)
    ax.yaxis.label.set_color(DIM_T)


ax1 = fig.add_subplot(gs[0, 0])
ax_style(ax1, '① Train vs Val Loss')
ep = range(1, len(history.history['loss'])+1)
ax1.plot(ep, history.history['loss'],     color=BLUC, lw=1.5, label='Train')
ax1.plot(ep, history.history['val_loss'],
         color=YELC, lw=1.5, label='Val', linestyle='--')
ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2 = fig.add_subplot(gs[0, 1])
ax_style(ax2, '② Train vs Val AUC')
ax2.plot(ep, history.history['auc'],     color=GRNC, lw=1.5, label='Train')
ax2.plot(ep, history.history['val_auc'], color=ORGC,
         lw=1.5, label='Val', linestyle='--')
ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')

ax3 = fig.add_subplot(gs[0, 2])
ax_style(ax3, f'③ ROC Curve (Test AUC = {auc_te:.3f})')
fpr_t, tpr_t, _ = roc_curve(yte, cal_te)
ax3.plot(fpr_t, tpr_t, color=GRNC, lw=2, label=f'Model (AUC={auc_te:.3f})')
ax3.plot([0, 1], [0, 1], color=GRID, lw=1, linestyle='--', label='Random')
ax3.fill_between(fpr_t, tpr_t, alpha=0.15, color=GRNC)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

ax4 = fig.add_subplot(gs[1, 0])
ax_style(ax4, '④ Precision–Recall Curve')
prec_p, rec_p, _ = precision_recall_curve(yte, cal_te)
ax4.plot(rec_p, prec_p, color=BLUC, lw=2)
ax4.axhline(0.5, color=GRID, lw=1, linestyle='--', label='Random')
ax4.axvline(THR, color=YELC, lw=1, linestyle=':',
            alpha=0.6, label=f'Base Center={THR:.2f}')
ax4.fill_between(rec_p, prec_p, alpha=0.1, color=BLUC)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

ax5 = fig.add_subplot(gs[1, 1])
ax_style(ax5, '⑤ Confidence Distribution (Test)')
bins_h = np.linspace(0, 1, 30)
ax5.hist(cal_te[yte == 0], bins=bins_h,
         alpha=0.6, color=REDC, label='Down days')
ax5.hist(cal_te[yte == 1], bins=bins_h, alpha=0.6, color=GRNC, label='Up days')
ax5.axvline(THR, color=YELC, lw=2, linestyle='--',
            label=f'Base Center={THR:.2f}')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Count')
ax5.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

ax6 = fig.add_subplot(gs[1, 2])
ax_style(ax6, '⑥ Backtest: Strategy vs Buy & Hold')
ax6.plot(range(BACKTEST_D), cum_strat*100, color=GRNC,
         lw=2, label='Strategy', marker='o', markersize=3)
ax6.plot(range(BACKTEST_D), cum_bh*100,    color=BLUC,
         lw=2, label='Buy & Hold', linestyle='--')
ax6.axhline(0, color=GRID, lw=1)
ax6.fill_between(range(BACKTEST_D), cum_strat*100, alpha=0.1, color=GRNC)
ax6.set_xlabel('Day')
ax6.set_ylabel('Return %')
ax6.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

ax7 = fig.add_subplot(gs[2, 0:2])
ax_style(ax7, '⑦ Backtest Daily Returns & Signals')
colors = [GRNC if s > 0 else (REDC if s < 0 else GRID) for s in sizes]
ax7.bar(range(BACKTEST_D), sr_daily*100, color=colors, alpha=0.7)
ax7.axhline(0, color=GRID, lw=1)

ax7b = ax7.twinx()
ax7b.plot(range(BACKTEST_D), bp*100, color=YELC,
          lw=1.5, alpha=0.8, label='Confidence')
ax7b.axhline(THR*100, color=YELC, lw=1, linestyle='--', alpha=0.4)
ax7b.set_ylabel('Confidence %', color=YELC, fontsize=7)
ax7b.tick_params(colors=YELC, labelsize=7)
ax7b.set_ylim(0, 100)
ax7.set_xlabel('Day')
ax7.set_ylabel('Return %')
patches = [Patch(color=GRNC, label='Long'), Patch(
    color=REDC, label='Short'), Patch(color=GRID, label='Cash')]
ax7.legend(handles=patches, fontsize=7, facecolor=PANEL, labelcolor=TEXT)

ax8 = fig.add_subplot(gs[2, 2])
ax_style(ax8, '⑧ Train / Val / Test Metrics')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
tr_vals = [m_tr[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
val_vals = [m_val[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
te_vals = [m_te[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
x = np.arange(len(metrics))
w = 0.25
ax8.bar(x-w, tr_vals,  w, color=BLUC, alpha=0.8, label='Train')
ax8.bar(x,   val_vals, w, color=YELC, alpha=0.8, label='Val')
ax8.bar(x+w, te_vals,  w, color=GRNC, alpha=0.8, label='Test')
ax8.set_xticks(x)
ax8.set_xticklabels(metrics, fontsize=7, rotation=15, color=DIM_T)
ax8.axhline(50, color=GRID, lw=1, linestyle='--')
ax8.set_ylim(0, 100)
ax8.set_ylabel('%')
ax8.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

fig.suptitle(f'BTC-USD LSTM Signal Engine  |  {datetime.now().strftime("%Y-%m-%d")}  |  '
             f'Test AUC: {auc_te:.3f}  |  Test Acc: {m_te["acc"]*100:.1f}%  |  '
             f'Strategy: {cum_strat[-1]*100:+.2f}%  |  Sharpe: {sh_strat:.2f}',
             color=TEXT, fontsize=10, fontweight='bold', y=0.98)

plt.show()
print("\n  Done. ✓\n")
