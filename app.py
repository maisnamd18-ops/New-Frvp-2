
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="FRVP Price Action Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NY = ZoneInfo("America/New_York")

# ============================================================
# Fixed strategy specification
# ============================================================
DEFAULT_ROWS = 60
DEFAULT_VA = 70.0
DEFAULT_TF = "5m"
DEFAULT_HISTORY = "30d"
DEFAULT_START_BALANCE = 5000.0
DEFAULT_RISK_PCT = 0.50
DEFAULT_SL_BUFFER = 0.20
DEFAULT_MAX_BARS = 60

# VAL / VAH bounce is ALWAYS exactly 2R.
BOUNCE_RR = 2.0

# POC logic is deliberately separate from the 2R bounce rule.
POC_BOUNCE_TARGET = "Opposite VA boundary"
POC_REVERSAL_RR = 1.0

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, interval: str, period: str):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(NY)
    df["Session"] = df.index.date
    return df.sort_index()

def frvp_profile(day_df, rows=60, value_area_pct=70.0):
    lo = float(day_df["Low"].min())
    hi = float(day_df["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan, None

    edges = np.linspace(lo, hi, rows + 1)
    vols = np.zeros(rows, dtype=float)

    # Approximate volume-at-price from OHLCV candles by distributing
    # each candle's volume uniformly across the price bins it overlaps.
    for _, r in day_df.iterrows():
        c_lo, c_hi = float(r["Low"]), float(r["High"])
        vol = float(r["Volume"]) if np.isfinite(r["Volume"]) else 0.0
        if vol <= 0 or c_hi <= c_lo:
            price = float(r["Close"])
            j = np.searchsorted(edges, price, side="right") - 1
            j = max(0, min(rows - 1, j))
            vols[j] += vol
            continue

        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], c_hi) - np.maximum(edges[:-1], c_lo)
        )
        total = overlap.sum()
        if total > 0:
            vols += vol * overlap / total

    poc_i = int(np.argmax(vols))
    total_vol = vols.sum()
    target = total_vol * (value_area_pct / 100.0)

    left = right = poc_i
    cum = vols[poc_i]
    while cum < target and (left > 0 or right < rows - 1):
        left_vol = vols[left - 1] if left > 0 else -1
        right_vol = vols[right + 1] if right < rows - 1 else -1
        if right_vol >= left_vol and right < rows - 1:
            right += 1
            cum += vols[right]
        elif left > 0:
            left -= 1
            cum += vols[left]
        else:
            break

    centers = (edges[:-1] + edges[1:]) / 2
    poc = float(centers[poc_i])
    val = float(edges[left])
    vah = float(edges[right + 1])
    profile = pd.DataFrame({"price": centers, "volume": vols})
    return poc, vah, val, profile

def build_session_levels(df, rows, va_pct):
    sessions = sorted(df["Session"].unique())
    levels = {}
    for i in range(1, len(sessions)):
        prev_s = sessions[i - 1]
        cur_s = sessions[i]
        prev = df[df["Session"] == prev_s]
        if prev.empty:
            continue
        poc, vah, val, profile = frvp_profile(prev, rows, va_pct)
        levels[cur_s] = {
            "previous_session": prev_s,
            "POC": poc,
            "VAH": vah,
            "VAL": val,
            "profile": profile,
        }
    return levels

def candle_touches(row, level):
    return float(row["Low"]) <= level <= float(row["High"])

def run_backtest(
    df,
    levels,
    start_balance=5000.0,
    risk_pct=0.5,
    sl_buffer=0.20,
    max_bars=60,
    setup_filter="All",
):
    balance = float(start_balance)
    peak = balance
    max_dd = 0.0
    trades = []
    equity_points = []
    active = None

    data = df.copy()
    data["BarNo"] = np.arange(len(data))

    # Process each session independently. Previous-session FRVP levels
    # are frozen for the entire current session.
    for session, sidx in data.groupby("Session", sort=True).groups.items():
        sidx = list(sidx)
        if session not in levels:
            continue
        lv = levels[session]
        poc, vah, val = lv["POC"], lv["VAH"], lv["VAL"]
        session_df = data.loc[sidx]

        for pos, (ts, row) in enumerate(session_df.iterrows()):
            global_i = int(row["BarNo"])

            # Manage an open position first.
            if active is not None:
                side = active["side"]
                hit_sl = float(row["Low"]) <= active["sl"] if side == "LONG" else float(row["High"]) >= active["sl"]
                hit_tp = float(row["High"]) >= active["tp"] if side == "LONG" else float(row["Low"]) <= active["tp"]

                exit_reason = None
                exit_price = None
                # Conservative OHLC rule: SL first if both occur in one candle.
                if hit_sl:
                    exit_reason, exit_price = "SL", active["sl"]
                elif hit_tp:
                    exit_reason, exit_price = "TP", active["tp"]
                elif global_i - active["entry_bar"] >= max_bars:
                    exit_reason, exit_price = "TIME", float(row["Close"])

                if exit_reason:
                    pnl = (exit_price - active["entry"]) * active["qty"]
                    if side == "SHORT":
                        pnl = -pnl
                    r = pnl / active["risk_money"] if active["risk_money"] else np.nan
                    balance += pnl
                    peak = max(peak, balance)
                    dd = peak - balance
                    max_dd = max(max_dd, dd)
                    trades.append({
                        "Entry Time": active["entry_time"],
                        "Exit Time": ts,
                        "Setup": active["setup"],
                        "Side": side,
                        "Entry": active["entry"],
                        "SL": active["sl"],
                        "TP": active["tp"],
                        "Exit": exit_price,
                        "Result": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
                        "R": r,
                        "PnL ($)": pnl,
                        "Balance ($)": balance,
                        "Previous Session": str(active["previous_session"]),
                        "POC": poc,
                        "VAH": vah,
                        "VAL": val,
                        "Exit Reason": exit_reason,
                    })
                    active = None

            equity_points.append((ts, balance))
            if active is not None:
                continue

            # Avoid signals on the final candle if there is no following bar.
            if pos >= len(session_df) - 1:
                continue

            o, h, l, c = map(float, [row["Open"], row["High"], row["Low"], row["Close"]])

            candidates = []

            # --------------------------------------------------------
            # VAL Bounce: bullish rejection from VAL, target EXACTLY 2R
            # --------------------------------------------------------
            if setup_filter in ("All", "VAL Bounce"):
                if l <= val and c > val and c > o:
                    entry = c
                    sl = l - sl_buffer
                    risk_price = entry - sl
                    if risk_price > 0:
                        tp = entry + BOUNCE_RR * risk_price
                        candidates.append(("VAL Bounce", "LONG", entry, sl, tp))

            # --------------------------------------------------------
            # VAH Bounce: bearish rejection from VAH, target EXACTLY 2R
            # --------------------------------------------------------
            if setup_filter in ("All", "VAH Bounce"):
                if h >= vah and c < vah and c < o:
                    entry = c
                    sl = h + sl_buffer
                    risk_price = sl - entry
                    if risk_price > 0:
                        tp = entry - BOUNCE_RR * risk_price
                        candidates.append(("VAH Bounce", "SHORT", entry, sl, tp))

            # --------------------------------------------------------
            # POC Bounce: retained separately from the 2R bounce rule.
            # Long from POC toward VAH; short from POC toward VAL.
            # --------------------------------------------------------
            if setup_filter in ("All", "POC Bounce"):
                if l <= poc <= h:
                    if c > poc and c > o:
                        entry = c
                        sl = l - sl_buffer
                        tp = vah
                        if entry > sl and tp > entry:
                            candidates.append(("POC Bounce", "LONG", entry, sl, tp))
                    elif c < poc and c < o:
                        entry = c
                        sl = h + sl_buffer
                        tp = val
                        if sl > entry and tp < entry:
                            candidates.append(("POC Bounce", "SHORT", entry, sl, tp))

            # --------------------------------------------------------
            # POC Reversal: cross through POC and close on the new side.
            # Target is 1R (unchanged from the previous POC-style rule).
            # --------------------------------------------------------
            if setup_filter in ("All", "POC Reversal"):
                prev_pos = pos - 1
                if prev_pos >= 0:
                    prev = session_df.iloc[prev_pos]
                    prev_c = float(prev["Close"])
                    if prev_c < poc and c > poc and c > o:
                        entry = c
                        sl = l - sl_buffer
                        risk_price = entry - sl
                        if risk_price > 0:
                            tp = entry + POC_REVERSAL_RR * risk_price
                            candidates.append(("POC Reversal", "LONG", entry, sl, tp))
                    elif prev_c > poc and c < poc and c < o:
                        entry = c
                        sl = h + sl_buffer
                        risk_price = sl - entry
                        if risk_price > 0:
                            tp = entry - POC_REVERSAL_RR * risk_price
                            candidates.append(("POC Reversal", "SHORT", entry, sl, tp))

            # --------------------------------------------------------
            # Breakout: close beyond VAH/VAL, then retest within the
            # value-area width tolerance. Target remains 1R by default.
            # --------------------------------------------------------
            if setup_filter in ("All", "VAH Breakout", "VAL Breakout"):
                if pos >= 1:
                    prev = session_df.iloc[pos - 1]
                    prev_c = float(prev["Close"])
                    width = max(vah - val, 1e-9)
                    tol = width * 0.08

                    if setup_filter in ("All", "VAH Breakout"):
                        if prev_c > vah and l <= vah + tol and c > vah:
                            entry = c
                            sl = l - sl_buffer
                            risk_price = entry - sl
                            if risk_price > 0:
                                tp = entry + risk_price
                                candidates.append(("VAH Breakout", "LONG", entry, sl, tp))

                    if setup_filter in ("All", "VAL Breakout"):
                        if prev_c < val and h >= val - tol and c < val:
                            entry = c
                            sl = h + sl_buffer
                            risk_price = sl - entry
                            if risk_price > 0:
                                tp = entry - risk_price
                                candidates.append(("VAL Breakout", "SHORT", entry, sl, tp))

            if not candidates:
                continue

            # If multiple setups fire on one candle, use deterministic priority.
            priority = {
                "VAL Bounce": 1,
                "VAH Bounce": 1,
                "POC Reversal": 2,
                "POC Bounce": 3,
                "VAH Breakout": 4,
                "VAL Breakout": 4,
            }
            candidates.sort(key=lambda x: priority.get(x[0], 99))
            setup, side, entry, sl, tp = candidates[0]

            risk_price = abs(entry - sl)
            risk_money = balance * risk_pct / 100.0
            qty = risk_money / risk_price if risk_price > 0 else 0.0

            if qty <= 0:
                continue

            active = {
                "entry_time": ts,
                "entry_bar": global_i,
                "setup": setup,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "risk_money": risk_money,
                "previous_session": lv["previous_session"],
            }

    return pd.DataFrame(trades), pd.Series(
        [v for _, v in equity_points],
        index=[t for t, _ in equity_points],
        dtype=float
    ), max_dd

def metrics(trades, start_balance):
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_r": 0.0, "avg_r": 0.0, "profit_factor": 0.0,
            "net_profit": 0.0, "ending_balance": start_balance,
            "max_consecutive_losses": 0
        }
    wins = int((trades["Result"] == "WIN").sum())
    losses = int((trades["Result"] == "LOSS").sum())
    gross_profit = trades.loc[trades["PnL ($)"] > 0, "PnL ($)"].sum()
    gross_loss = -trades.loc[trades["PnL ($)"] < 0, "PnL ($)"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    seq = 0
    max_seq = 0
    for r in trades["Result"]:
        if r == "LOSS":
            seq += 1
            max_seq = max(max_seq, seq)
        else:
            seq = 0
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100,
        "net_r": trades["R"].sum(),
        "avg_r": trades["R"].mean(),
        "profit_factor": pf,
        "net_profit": trades["PnL ($)"].sum(),
        "ending_balance": float(trades["Balance ($)"].iloc[-1]),
        "max_consecutive_losses": max_seq,
    }

# ============================================================
# UI
# ============================================================
st.title("FRVP Price Action Backtester")
st.caption("Previous-session Fixed Range Volume Profile • Price Action only • XAUUSD/Gold")

with st.sidebar:
    st.header("FRVP")
    rows = st.number_input("Row Size", 20, 200, DEFAULT_ROWS, 5, key="frvp_rows_v8")
    va_pct = st.number_input("Value Area %", 50.0, 90.0, DEFAULT_VA, 1.0, key="frvp_va_v8")
    st.caption("FRVP range = the entire previous completed trading session.")

    st.header("Market Data")
    symbol = st.selectbox(
        "Instrument",
        ["XAUUSD=X", "GC=F"],
        index=0,
        key="symbol_v8",
        help="XAUUSD=X = Yahoo spot gold. GC=F = COMEX gold futures."
    )
    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m"],
        index=0,
        key="tf_v8"
    )
    history = st.selectbox(
        "Historical data",
        ["5d", "10d", "20d", "30d", "60d"],
        index=3,
        key="history_v8"
    )

    st.header("Strategy")
    setup_filter = st.selectbox(
        "Setup",
        ["All", "VAL Bounce", "VAH Bounce", "POC Bounce", "POC Reversal", "VAH Breakout", "VAL Breakout"],
        index=0,
        key="setup_v8"
    )

    st.header("Trade Rules")
    sl_buffer = st.number_input(
        "SL buffer ($)",
        0.0, 5.0, DEFAULT_SL_BUFFER, 0.05,
        key="sl_buffer_v8"
    )
    max_bars = st.number_input(
        "Maximum bars in trade",
        5, 300, DEFAULT_MAX_BARS, 5,
        key="max_bars_v8"
    )

    st.header("Balance & Risk")
    start_balance = st.number_input(
        "Starting Balance ($)",
        100.0, 1_000_000.0, DEFAULT_START_BALANCE, 100.0,
        key="balance_v8"
    )
    risk_pct = st.number_input(
        "Risk per trade (%)",
        0.1, 5.0, DEFAULT_RISK_PCT, 0.1,
        key="risk_v8"
    )
    st.caption("Position size is calculated automatically from balance and stop distance.")

    run = st.button("🔄 Fetch Data & Run Backtest", type="primary", use_container_width=True, key="run_v8")

if run:
    st.session_state["run_v8_once"] = True

if st.session_state.get("run_v8_once"):
    with st.spinner("Fetching Gold data and calculating previous-session FRVP..."):
        df = load_data(symbol, timeframe, history)

    if df.empty:
        st.error("Automatic Gold data could not be loaded. Try another timeframe or historical period.")
        st.stop()

    levels = build_session_levels(df, int(rows), float(va_pct))
    trades, equity, max_dd = run_backtest(
        df, levels,
        start_balance=float(start_balance),
        risk_pct=float(risk_pct),
        sl_buffer=float(sl_buffer),
        max_bars=int(max_bars),
        setup_filter=setup_filter,
    )
    m = metrics(trades, float(start_balance))

    st.success(f"Loaded {len(df):,} Gold candles automatically from Yahoo Finance.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Win Rate", f"{m['win_rate']:.1f}%")
    c2.metric("Trades", f"{m['trades']}")
    c3.metric("Net R", f"{m['net_r']:.2f}R")
    c4.metric("Avg R", f"{m['avg_r']:.2f}R")
    c5.metric("Profit Factor", "∞" if np.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
    c6.metric("Ending Balance", f"${m['ending_balance']:,.2f}")

    st.subheader("Strategy Rules")
    st.info(
        f"VAL Bounce and VAH Bounce targets are hard-coded to exactly {BOUNCE_RR:.0f}R. "
        f"POC logic is kept separate: POC Bounce targets the opposite value-area boundary; "
        f"POC Reversal uses {POC_REVERSAL_RR:.0f}R. "
        f"FRVP is calculated once from the entire previous completed session and then frozen for the current session."
    )

    if not trades.empty:
        st.subheader("Performance by Setup")
        stats = trades.groupby("Setup").agg(
            Trades=("Setup", "size"),
            Wins=("Result", lambda x: (x == "WIN").sum()),
            Losses=("Result", lambda x: (x == "LOSS").sum()),
            Net_R=("R", "sum"),
            Avg_R=("R", "mean"),
            PnL=("PnL ($)", "sum"),
        ).reset_index()
        stats["Win Rate %"] = stats["Wins"] / stats["Trades"] * 100
        stats = stats[["Setup", "Trades", "Wins", "Losses", "Win Rate %", "Net_R", "Avg_R", "PnL"]]
        st.dataframe(stats.round(2), use_container_width=True, hide_index=True)

        st.subheader("Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Balance"))
        fig_eq.update_layout(height=320, yaxis_title="Balance ($)", xaxis_title="Time")
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Trade Log")
        display = trades.copy()
        for col in ["Entry", "SL", "TP", "Exit", "R", "PnL ($)", "Balance ($)", "POC", "VAH", "VAL"]:
            if col in display.columns:
                display[col] = display[col].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Trade Log CSV",
            trades.to_csv(index=False).encode("utf-8"),
            "frvp_trade_log_v8.csv",
            "text/csv",
            use_container_width=True,
            key="download_trades_v8"
        )
    else:
        st.warning("No completed trades matched the selected setup/settings.")

    st.subheader("Session FRVP Levels")
    rows_out = []
    for session, lv in levels.items():
        rows_out.append({
            "Current Session": str(session),
            "Previous Session": str(lv["previous_session"]),
            "POC": lv["POC"],
            "VAH": lv["VAH"],
            "VAL": lv["VAL"],
        })
    if rows_out:
        st.dataframe(pd.DataFrame(rows_out).round(2), use_container_width=True, hide_index=True)

    st.caption(
        f"Max drawdown: ${max_dd:,.2f}. "
        "Backtest uses OHLCV data and conservatively counts SL first when SL and TP occur in the same candle. "
        "Spread, commission, slippage and broker-specific contract specifications are not modeled."
    )
else:
    st.info(
        "Press “Fetch Data & Run Backtest”. No CSV upload is required. "
        "The app automatically fetches Gold data and builds a separate previous-session FRVP for every trading session."
    )
