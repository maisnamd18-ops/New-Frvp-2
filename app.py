
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="FRVP Price Action Backtester V9",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NY = ZoneInfo("America/New_York")

# ============================================================
# DEFAULTS — balanced baseline
# ============================================================
ROWS = 60
VA_PCT = 70.0
TIMEFRAME = "5m"
HISTORY = "30d"
START_BALANCE = 5000.0
RISK_PCT = 0.50
SL_BUFFER = 0.20
MAX_BARS = 36
BOUNCE_RR = 2.0
BREAKOUT_RR = 1.0

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_gold(interval, period):
    # Yahoo may return 404/no-data for XAUUSD=X. Try COMEX gold futures
    # first, then fall back to the spot symbol. GC=F is not broker-specific
    # spot XAUUSD; it is Yahoo's Gold Futures feed.
    for symbol in ("GC=F", "XAUUSD=X"):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in cols):
                continue

            df = df[cols].copy().dropna(subset=["Open", "High", "Low", "Close"])
            if df.empty:
                continue

            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df.index = idx.tz_convert(NY)
            df["Session"] = df.index.date
            df.attrs["source_symbol"] = symbol
            return df.sort_index()
        except Exception:
            continue

    return pd.DataFrame()


def frvp_profile(day_df, rows=60, value_area_pct=70.0):
    lo = float(day_df["Low"].min())
    hi = float(day_df["High"].max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan

    edges = np.linspace(lo, hi, rows + 1)
    vols = np.zeros(rows, dtype=float)

    # OHLCV volume-at-price approximation.
    for _, r in day_df.iterrows():
        low = float(r["Low"])
        high = float(r["High"])
        vol = float(r["Volume"]) if np.isfinite(r["Volume"]) else 0.0

        if vol <= 0:
            continue

        if high <= low:
            j = np.searchsorted(edges, float(r["Close"]), side="right") - 1
            j = max(0, min(rows - 1, j))
            vols[j] += vol
            continue

        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
        )
        total = overlap.sum()
        if total > 0:
            vols += vol * overlap / total

    poc_i = int(np.argmax(vols))
    total = vols.sum()
    target = total * value_area_pct / 100.0

    left = right = poc_i
    cumulative = vols[poc_i]

    while cumulative < target and (left > 0 or right < rows - 1):
        lv = vols[left - 1] if left > 0 else -1
        rv = vols[right + 1] if right < rows - 1 else -1

        if rv >= lv and right < rows - 1:
            right += 1
            cumulative += vols[right]
        elif left > 0:
            left -= 1
            cumulative += vols[left]
        else:
            break

    centers = (edges[:-1] + edges[1:]) / 2
    poc = float(centers[poc_i])
    val = float(edges[left])
    vah = float(edges[right + 1])
    return poc, vah, val


def build_previous_day_levels(df, rows=60, va_pct=70.0):
    sessions = sorted(df["Session"].unique())
    levels = {}

    for i in range(1, len(sessions)):
        previous_day = sessions[i - 1]
        current_day = sessions[i]

        previous = df[df["Session"] == previous_day]
        if previous.empty:
            continue

        poc, vah, val = frvp_profile(previous, rows, va_pct)

        levels[current_day] = {
            "Previous Day": previous_day,
            "POC": poc,
            "VAH": vah,
            "VAL": val,
        }

    return levels


# ============================================================
# PRICE-ACTION HELPERS
# ============================================================
def body(o, c):
    return abs(c - o)


def candle_range(h, l):
    return max(h - l, 1e-9)


def bullish_rejection(o, h, l, c, level):
    """
    Balanced VAL bounce:
    - price trades through/touches VAL
    - closes back above VAL
    - bullish candle
    - lower wick is meaningful relative to body/range
    """
    b = body(o, c)
    r = candle_range(h, l)
    lower_wick = min(o, c) - l

    return (
        l <= level
        and c > level
        and c > o
        and lower_wick >= max(b * 0.75, r * 0.20)
    )


def bearish_rejection(o, h, l, c, level):
    """
    Balanced VAH bounce:
    - price trades through/touches VAH
    - closes back below VAH
    - bearish candle
    - upper wick is meaningful
    """
    b = body(o, c)
    r = candle_range(h, l)
    upper_wick = h - max(o, c)

    return (
        h >= level
        and c < level
        and c < o
        and upper_wick >= max(b * 0.75, r * 0.20)
    )


def run_backtest(
    df,
    levels,
    setup_filter="All",
    start_balance=5000.0,
    risk_pct=0.50,
    sl_buffer=0.20,
    max_bars=36,
):
    balance = float(start_balance)
    peak = balance
    max_dd = 0.0

    trades = []
    equity = []
    used_levels = set()

    data = df.copy()
    data["BarNo"] = np.arange(len(data))

    for session, group in data.groupby("Session", sort=True):
        if session not in levels:
            continue

        lv = levels[session]
        poc = lv["POC"]
        vah = lv["VAH"]
        val = lv["VAL"]

        g = group.copy()
        active = None

        for pos in range(len(g)):
            ts = g.index[pos]
            row = g.iloc[pos]
            global_bar = int(row["BarNo"])

            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
            c = float(row["Close"])

            # ----------------------------------------------------
            # Manage open trade first.
            # ----------------------------------------------------
            if active is not None:
                side = active["side"]

                hit_sl = (
                    l <= active["sl"] if side == "LONG"
                    else h >= active["sl"]
                )
                hit_tp = (
                    h >= active["tp"] if side == "LONG"
                    else l <= active["tp"]
                )

                exit_reason = None
                exit_price = None

                # Conservative OHLC assumption: SL first if both
                # SL and TP are inside the same candle.
                if hit_sl:
                    exit_reason = "SL"
                    exit_price = active["sl"]
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = active["tp"]
                elif global_bar - active["entry_bar"] >= max_bars:
                    exit_reason = "TIME"
                    exit_price = c

                if exit_reason:
                    pnl = (
                        (exit_price - active["entry"]) * active["qty"]
                        if side == "LONG"
                        else (active["entry"] - exit_price) * active["qty"]
                    )

                    r = pnl / active["risk_money"]
                    balance += pnl
                    peak = max(peak, balance)
                    max_dd = max(max_dd, peak - balance)

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
                        "Previous Day": str(active["previous_day"]),
                        "POC": poc,
                        "VAH": vah,
                        "VAL": val,
                        "Exit Reason": exit_reason,
                    })

                    # A level can be used again only on a fresh setup,
                    # but not repeatedly on every candle after the same touch.
                    active = None

            equity.append((ts, balance))

            if active is not None:
                continue

            # No new entry on the last candle of the session.
            if pos >= len(g) - 1:
                continue

            prev = g.iloc[pos - 1] if pos > 0 else None
            candidates = []

            # ====================================================
            # 1. VAL BOUNCE — EXACTLY 2R
            # ====================================================
            if setup_filter in ("All", "VAL Bounce"):
                key = (session, "VAL Bounce")
                if key not in used_levels and bullish_rejection(o, h, l, c, val):
                    entry = c
                    sl = l - sl_buffer
                    risk = entry - sl

                    if risk > 0:
                        tp = entry + BOUNCE_RR * risk
                        candidates.append(
                            ("VAL Bounce", "LONG", entry, sl, tp, key)
                        )

            # ====================================================
            # 2. VAH BOUNCE — EXACTLY 2R
            # ====================================================
            if setup_filter in ("All", "VAH Bounce"):
                key = (session, "VAH Bounce")
                if key not in used_levels and bearish_rejection(o, h, l, c, vah):
                    entry = c
                    sl = h + sl_buffer
                    risk = sl - entry

                    if risk > 0:
                        tp = entry - BOUNCE_RR * risk
                        candidates.append(
                            ("VAH Bounce", "SHORT", entry, sl, tp, key)
                        )

            # ====================================================
            # 3. POC BOUNCE — LOGIC RETAINED
            # ====================================================
            if setup_filter in ("All", "POC Bounce"):
                key = (session, "POC Bounce")

                if key not in used_levels and l <= poc <= h:
                    if c > poc and c > o:
                        entry = c
                        sl = l - sl_buffer
                        tp = vah

                        if entry > sl and tp > entry:
                            candidates.append(
                                ("POC Bounce", "LONG", entry, sl, tp, key)
                            )

                    elif c < poc and c < o:
                        entry = c
                        sl = h + sl_buffer
                        tp = val

                        if sl > entry and tp < entry:
                            candidates.append(
                                ("POC Bounce", "SHORT", entry, sl, tp, key)
                            )

            # ====================================================
            # 4. POC REVERSAL — LOGIC RETAINED
            # ====================================================
            if setup_filter in ("All", "POC Reversal"):
                key = (session, "POC Reversal")

                if key not in used_levels and prev is not None:
                    prev_close = float(prev["Close"])

                    if prev_close < poc and c > poc and c > o:
                        entry = c
                        sl = l - sl_buffer
                        risk = entry - sl

                        if risk > 0:
                            tp = entry + risk
                            candidates.append(
                                ("POC Reversal", "LONG", entry, sl, tp, key)
                            )

                    elif prev_close > poc and c < poc and c < o:
                        entry = c
                        sl = h + sl_buffer
                        risk = sl - entry

                        if risk > 0:
                            tp = entry - risk
                            candidates.append(
                                ("POC Reversal", "SHORT", entry, sl, tp, key)
                            )

            # ====================================================
            # 5. VAH BREAKOUT — CLOSE + RETEST + CONTINUATION
            #    One signal per VAH per session.
            # ====================================================
            if setup_filter in ("All", "VAH Breakout"):
                key = (session, "VAH Breakout")

                if key not in used_levels and prev is not None:
                    prev_close = float(prev["Close"])
                    width = max(vah - val, 1e-9)
                    tol = width * 0.08

                    if prev_close > vah and l <= vah + tol and c > vah and c > o:
                        entry = c
                        sl = l - sl_buffer
                        risk = entry - sl

                        if risk > 0:
                            tp = entry + BREAKOUT_RR * risk
                            candidates.append(
                                ("VAH Breakout", "LONG", entry, sl, tp, key)
                            )

            # ====================================================
            # 6. VAL BREAKOUT — CLOSE + RETEST + CONTINUATION
            # ====================================================
            if setup_filter in ("All", "VAL Breakout"):
                key = (session, "VAL Breakout")

                if key not in used_levels and prev is not None:
                    prev_close = float(prev["Close"])
                    width = max(vah - val, 1e-9)
                    tol = width * 0.08

                    if prev_close < val and h >= val - tol and c < val and c < o:
                        entry = c
                        sl = h + sl_buffer
                        risk = sl - entry

                        if risk > 0:
                            tp = entry - BREAKOUT_RR * risk
                            candidates.append(
                                ("VAL Breakout", "SHORT", entry, sl, tp, key)
                            )

            if not candidates:
                continue

            # Priority: boundary rejection first, then POC, then breakout.
            priority = {
                "VAL Bounce": 1,
                "VAH Bounce": 1,
                "POC Reversal": 2,
                "POC Bounce": 3,
                "VAH Breakout": 4,
                "VAL Breakout": 4,
            }
            candidates.sort(key=lambda x: priority[x[0]])

            setup, side, entry, sl, tp, key = candidates[0]

            risk_price = abs(entry - sl)
            risk_money = balance * risk_pct / 100.0
            qty = risk_money / risk_price if risk_price > 0 else 0

            if qty <= 0:
                continue

            active = {
                "entry_time": ts,
                "entry_bar": global_bar,
                "setup": setup,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "risk_money": risk_money,
                "previous_day": lv["Previous Day"],
            }

            used_levels.add(key)

        # Force-close an open trade at the final available candle of
        # the session so no position leaks into the next day.
        if active is not None:
            last_ts = g.index[-1]
            last_close = float(g.iloc[-1]["Close"])
            side = active["side"]

            pnl = (
                (last_close - active["entry"]) * active["qty"]
                if side == "LONG"
                else (active["entry"] - last_close) * active["qty"]
            )
            r = pnl / active["risk_money"]
            balance += pnl
            peak = max(peak, balance)
            max_dd = max(max_dd, peak - balance)

            trades.append({
                "Entry Time": active["entry_time"],
                "Exit Time": last_ts,
                "Setup": active["setup"],
                "Side": side,
                "Entry": active["entry"],
                "SL": active["sl"],
                "TP": active["tp"],
                "Exit": last_close,
                "Result": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
                "R": r,
                "PnL ($)": pnl,
                "Balance ($)": balance,
                "Previous Day": str(active["previous_day"]),
                "POC": poc,
                "VAH": vah,
                "VAL": val,
                "Exit Reason": "SESSION_CLOSE",
            })

            equity.append((last_ts, balance))

    trades_df = pd.DataFrame(trades)

    if equity:
        eq = pd.Series(
            [v for _, v in equity],
            index=[t for t, _ in equity],
            dtype=float,
        )
    else:
        eq = pd.Series(dtype=float)

    return trades_df, eq, max_dd


def metrics(trades, start_balance):
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_r": 0, "avg_r": 0, "pf": 0,
            "ending_balance": start_balance,
            "max_loss_streak": 0,
        }

    wins = int((trades["Result"] == "WIN").sum())
    losses = int((trades["Result"] == "LOSS").sum())

    gp = trades.loc[trades["PnL ($)"] > 0, "PnL ($)"].sum()
    gl = -trades.loc[trades["PnL ($)"] < 0, "PnL ($)"].sum()
    pf = gp / gl if gl > 0 else np.inf

    streak = max_streak = 0
    for x in trades["Result"]:
        if x == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100,
        "net_r": trades["R"].sum(),
        "avg_r": trades["R"].mean(),
        "pf": pf,
        "ending_balance": float(trades["Balance ($)"].iloc[-1]),
        "max_loss_streak": max_streak,
    }


# ============================================================
# UI
# ============================================================
st.title("FRVP Price Action Backtester V10")
st.caption(
    "Previous-day FRVP • Price action confirmation • "
    "VAL/VAH bounce = exactly 2R • POC logic unchanged"
)

with st.sidebar:
    st.header("FRVP")
    rows = st.number_input(
        "Row Size", min_value=20, max_value=200, value=ROWS, step=5,
        key="v10_rows"
    )
    va_pct = st.number_input(
        "Value Area %", min_value=50.0, max_value=90.0, value=VA_PCT, step=1.0,
        key="v10_va"
    )
    st.caption("Every trading day uses only the immediately preceding completed day.")

    st.header("Market Data")
    timeframe = st.selectbox(
        "Timeframe", ["5m", "15m"], index=0, key="v10_tf"
    )
    history = st.selectbox(
        "Historical data", ["5d", "10d", "20d", "30d", "60d"],
        index=3, key="v10_history"
    )

    st.header("Strategy")
    setup = st.selectbox(
        "Setup",
        ["All", "VAL Bounce", "VAH Bounce", "POC Bounce",
         "POC Reversal", "VAH Breakout", "VAL Breakout"],
        index=0, key="v10_setup"
    )

    st.header("Trade Rules")
    st.number_input(
        "Bounce Target (R)",
        min_value=2.0, max_value=2.0, value=2.0, step=0.5,
        disabled=True, key="v10_bounce_rr"
    )
    sl_buffer = st.number_input(
        "SL Buffer ($)", min_value=0.0, max_value=5.0,
        value=SL_BUFFER, step=0.05, key="v10_sl"
    )
    max_bars = st.number_input(
        "Maximum bars in trade", min_value=5, max_value=300,
        value=MAX_BARS, step=5, key="v10_maxbars"
    )

    st.header("Balance & Risk")
    start_balance = st.number_input(
        "Starting Balance ($)", min_value=100.0,
        max_value=1000000.0, value=START_BALANCE, step=100.0,
        key="v10_balance"
    )
    risk_pct = st.number_input(
        "Risk per trade (%)", min_value=0.1, max_value=5.0,
        value=RISK_PCT, step=0.1, key="v10_risk"
    )

    run = st.button(
        "🔄 Fetch Gold & Run Backtest",
        type="primary",
        use_container_width=True,
        key="v10_run"
    )

if run:
    st.session_state["v10_run_once"] = True

if not st.session_state.get("v10_run_once"):
    st.info(
        "Press “Fetch Gold & Run Backtest”. No CSV upload is required. "
        "The app automatically downloads Gold data and creates a separate "
        "previous-day FRVP for every current day."
    )
    st.stop()

with st.spinner("Fetching Gold data and running V9 backtest..."):
    df = load_gold(timeframe, history)

if df.empty:
    st.error(
        "Automatic Gold data could not be loaded. Try 5m with 5d/10d or "
        "15m with a longer period, and check the Streamlit logs if Yahoo is unavailable."
    )
    st.stop()

source_symbol = df.attrs.get("source_symbol", "GC=F")
if source_symbol == "GC=F":
    st.info(
        "Automatic source: GC=F (COMEX Gold Futures). Yahoo's XAUUSD=X feed "
        "was unavailable, so the app used GC=F automatically. This is futures "
        "data, not broker-specific spot XAUUSD."
    )

levels = build_previous_day_levels(df, int(rows), float(va_pct))

trades, equity, max_dd = run_backtest(
    df=df,
    levels=levels,
    setup_filter=setup,
    start_balance=float(start_balance),
    risk_pct=float(risk_pct),
    sl_buffer=float(sl_buffer),
    max_bars=int(max_bars),
)

m = metrics(trades, float(start_balance))

st.success(f"Loaded {len(df):,} Gold candles automatically.")

c = st.columns(7)
c[0].metric("Win Rate", f"{m['win_rate']:.1f}%")
c[1].metric("Trades", str(m["trades"]))
c[2].metric("Net R", f"{m['net_r']:.2f}R")
c[3].metric("Avg R", f"{m['avg_r']:.2f}R")
c[4].metric("Profit Factor", "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}")
c[5].metric("Ending Balance", f"${m['ending_balance']:,.2f}")
c[6].metric("Max DD", f"${max_dd:,.2f}")

st.subheader("Exact Strategy Rules")
st.info(
    "1) FRVP is calculated separately for each completed day. "
    "2) The current day uses only the previous day's fixed POC/VAH/VAL. "
    "3) VAL/VAH bounce requires a real rejection candle and closes back inside value. "
    "4) VAL/VAH bounce TP is exactly 2R. "
    "5) POC Bounce and POC Reversal logic is retained separately. "
    "6) Breakout requires close outside the boundary, retest and continuation. "
    "7) Each setup is allowed only once per level per day to prevent repeated entries. "
    "8) If SL and TP are both inside one OHLC candle, SL is counted first."
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
    stats = stats[
        ["Setup", "Trades", "Wins", "Losses", "Win Rate %",
         "Net_R", "Avg_R", "PnL"]
    ]
    st.dataframe(stats.round(2), use_container_width=True, hide_index=True)

    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Balance",
        )
    )
    fig.update_layout(height=330, yaxis_title="Balance ($)", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trade Log")
    display = trades.copy()
    for col in ["Entry", "SL", "TP", "Exit", "R", "PnL ($)",
                "Balance ($)", "POC", "VAH", "VAL"]:
        if col in display.columns:
            display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Trade Log CSV",
        trades.to_csv(index=False).encode("utf-8"),
        "frvp_v9_trade_log.csv",
        "text/csv",
        use_container_width=True,
        key="v9_download"
    )
else:
    st.warning("No completed trades matched the selected settings.")

st.subheader("Previous-Day FRVP Levels")
level_rows = []
for current_day, lv in levels.items():
    level_rows.append({
        "Current Day": str(current_day),
        "Previous Day": str(lv["Previous Day"]),
        "POC": lv["POC"],
        "VAH": lv["VAH"],
        "VAL": lv["VAL"],
    })

if level_rows:
    st.dataframe(
        pd.DataFrame(level_rows).round(2),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Backtest is a research tool. Yahoo XAUUSD data may differ from a broker's "
    "feed; spread, commission and slippage are not modeled."
)
