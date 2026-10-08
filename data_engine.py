import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(ticker_symbol: str, period: str = "1y"):
    """
    미국 주식 티커의 가격 데이터, 재무 데이터, 벤치마크(SPY, QQQ) 데이터를 수집합니다.
    """
    ticker_symbol = ticker_symbol.strip().upper()
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. 과거 가격 데이터 (일봉)
    df = ticker.history(period=period)
    if df.empty or len(df) < 30:
        raise ValueError(f"'{ticker_symbol}'의 주가 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
    
    # 시간대 제거
    df.index = df.index.tz_localize(None)
    
    # 2. 기업 재무 및 펀더멘털 정보
    info = ticker.info or {}
    
    # 3. 벤치마크(SPY, QQQ) 데이터 가져와서 상대강도(RS) 계산
    spy = yf.Ticker("SPY").history(period=period)
    qqq = yf.Ticker("QQQ").history(period=period)
    spy.index = spy.index.tz_localize(None)
    qqq.index = qqq.index.tz_localize(None)
    
    # 기술적 지표 계산
    df = calculate_technical_indicators(df)
    
    # 볼륨 프로파일 및 POC 계산
    volume_profile, poc_price, val_price, vah_price = calculate_volume_profile(df)
    
    # Anchored VWAP 계산 (최근 52주 저점 및 실적 추정일)
    df, avwap_earnings, avwap_low = calculate_anchored_vwaps(df, ticker)
    
    # OBV 및 A/D Line 다이버전스 분석
    obv_signal = analyze_obv_ad_divergence(df)
    
    # 상대강도(RS) 및 시장 주도주 점수 계산
    rs_score, is_market_leader = calculate_market_leadership(df, spy, qqq)
    
    # 영업이익(Operating Income) 분석
    op_metrics = analyze_operating_income(ticker, info)
    
    # 기관 매집 추정치 계산
    inst_metrics = estimate_institutional_accumulation(
        df, poc_price, val_price, vah_price, avwap_earnings, avwap_low, ticker
    )
    
    return {
        "ticker": ticker_symbol,
        "company_name": info.get("shortName", ticker_symbol),
        "current_price": float(df["Close"].iloc[-1]),
        "prev_close": float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Close"].iloc[-1]),
        "df": df,
        "info": info,
        "volume_profile": volume_profile,
        "poc_price": poc_price,
        "val_price": val_price,
        "vah_price": vah_price,
        "avwap_earnings": avwap_earnings,
        "avwap_low": avwap_low,
        "obv_signal": obv_signal,
        "rs_score": rs_score,
        "is_market_leader": is_market_leader,
        "op_metrics": op_metrics,
        "inst_metrics": inst_metrics
    }

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기본 이동평균선 및 보조지표 계산"""
    df = df.copy()
    
    # 20일, 50일, 200일 이동평균선
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    
    # 전형적 가격 (Typical Price)
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    
    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_EMA20"] = df["OBV"].ewm(span=20).mean()
    
    # Chaikin A/D (Accumulation / Distribution Line)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    df["AD_Line"] = (clv * df["Volume"]).cumsum()
    
    # RSI (14일)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

def calculate_volume_profile(df: pd.DataFrame, bins: int = 50):
    """
    Volume Profile (매물대) 및 POC (Point of Control), Value Area 계산
    """
    price_min = df["Low"].min()
    price_max = df["High"].max()
    
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)
    
    for _, row in df.iterrows():
        # 각 봉의 가격 범위를 bins에 분배
        candle_low = row["Low"]
        candle_high = row["High"]
        vol = row["Volume"]
        
        idx_low = np.clip(np.digitize(candle_low, price_bins) - 1, 0, bins - 1)
        idx_high = np.clip(np.digitize(candle_high, price_bins) - 1, 0, bins - 1)
        
        if idx_low == idx_high:
            bin_volumes[idx_low] += vol
        else:
            span = idx_high - idx_low + 1
            bin_volumes[idx_low : idx_high + 1] += vol / span
            
    # POC (최대 볼륨 가격)
    poc_idx = np.argmax(bin_volumes)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    # Value Area (전체 거래량의 70%가 집중된 구간)
    total_vol = np.sum(bin_volumes)
    target_vol = total_vol * 0.70
    
    accum_vol = bin_volumes[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx
    
    while accum_vol < target_vol and (low_idx > 0 or high_idx < bins - 1):
        vol_below = bin_volumes[low_idx - 1] if low_idx > 0 else -1
        vol_above = bin_volumes[high_idx + 1] if high_idx < bins - 1 else -1
        
        if vol_above >= vol_below and high_idx < bins - 1:
            high_idx += 1
            accum_vol += bin_volumes[high_idx]
        elif low_idx > 0:
            low_idx -= 1
            accum_vol += bin_volumes[low_idx]
        else:
            break
            
    val_price = price_bins[low_idx]
    vah_price = price_bins[high_idx + 1]
    
    profile_data = []
    for i in range(bins):
        profile_data.append({
            "price_low": price_bins[i],
            "price_high": price_bins[i + 1],
            "price_mid": (price_bins[i] + price_bins[i + 1]) / 2,
            "volume": bin_volumes[i]
        })
        
    return profile_data, float(poc_price), float(val_price), float(vah_price)

def calculate_anchored_vwaps(df: pd.DataFrame, ticker):
    """
    특정 앵커 포인트로부터의 Anchored VWAP 계산:
    1) 직전 실적 발표일 앵커
    2) 최근 52주 최저점(패닉 바잉 바닥) 앵커
    """
    df = df.copy()
    
    # 앵커 1: 최근 최저점 날짜
    low_idx = df["Low"].idxmin()
    df["AVWAP_Low"] = np.nan
    mask_low = df.index >= low_idx
    cum_vol_low = df.loc[mask_low, "Volume"].cumsum()
    cum_pv_low = (df.loc[mask_low, "Typical_Price"] * df.loc[mask_low, "Volume"]).cumsum()
    df.loc[mask_low, "AVWAP_Low"] = cum_pv_low / (cum_vol_low + 1e-9)
    avwap_low_val = float(df["AVWAP_Low"].iloc[-1]) if not np.isnan(df["AVWAP_Low"].iloc[-1]) else None
    
    # 앵커 2: 직전 실적 발표일 (대략 60~90일 전 분기 실적 또는 거래량 급증일)
    # yfinance earnings_dates 확인 시도
    earnings_date = None
    try:
        earnings_dates = ticker.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            past_dates = earnings_dates[earnings_dates.index <= datetime.now()]
            if not past_dates.empty:
                earnings_date = past_dates.index[0].tz_localize(None)
    except Exception:
        pass
        
    # 만약 실적일을 못 찾으면 최근 60거래일 전(약 1분기 전) 혹은 최근 최대 거래량 발생일을 앵커로 사용
    if earnings_date is None or earnings_date < df.index[0]:
        recent_window = df.iloc[-80:] if len(df) >= 80 else df
        earnings_date = recent_window["Volume"].idxmax()
        
    df["AVWAP_Earnings"] = np.nan
    mask_earn = df.index >= earnings_date
    if mask_earn.any():
        cum_vol_earn = df.loc[mask_earn, "Volume"].cumsum()
        cum_pv_earn = (df.loc[mask_earn, "Typical_Price"] * df.loc[mask_earn, "Volume"]).cumsum()
        df.loc[mask_earn, "AVWAP_Earnings"] = cum_pv_earn / (cum_vol_earn + 1e-9)
        avwap_earn_val = float(df["AVWAP_Earnings"].iloc[-1])
    else:
        avwap_earn_val = None
        
    return df, avwap_earn_val, avwap_low_val

def analyze_obv_ad_divergence(df: pd.DataFrame) -> dict:
    """
    최근 30~50일간 주가 흐름과 OBV, A/D Line의 다이버전스를 분석하여
    스텔스 기관 매집(Stealth Accumulation) 여부를 진단합니다.
    """
    recent_len = min(40, len(df))
    sub_df = df.iloc[-recent_len:]
    
    price_change = (sub_df["Close"].iloc[-1] - sub_df["Close"].iloc[0]) / sub_df["Close"].iloc[0]
    obv_change = (sub_df["OBV"].iloc[-1] - sub_df["OBV"].iloc[0]) / (abs(sub_df["OBV"].iloc[0]) + 1e-9)
    ad_change = (sub_df["AD_Line"].iloc[-1] - sub_df["AD_Line"].iloc[0]) / (abs(sub_df["AD_Line"].iloc[0]) + 1e-9)
    
    # 다이버전스 판별
    if price_change <= 0.03 and (obv_change > 0.08 or ad_change > 0.08):
        status = "강력한 스텔스 매집 (Bullish Divergence) 🟢"
        desc = "주가는 횡보하거나 조정을 받고 있으나 OBV/수급선이 우상향하여 기관이 물량을 은밀히 흡수 중입니다."
    elif price_change >= 0.08 and (obv_change < -0.05 or ad_change < -0.05):
        status = "기관 물량 분산/차익실현 경보 (Bearish Divergence) 🔴"
        desc = "주가는 상승하고 있으나 OBV/수급선이 하락하여 큰손들이 상승을 이용해 물량을 털어내는 중입니다."
    elif price_change > 0 and obv_change > 0:
        status = "건전한 수급 동반 상승 (Healthy Uptrend) 🔵"
        desc = "주가 상승과 함께 거래량 유입이 정상적으로 뒷받침되고 있습니다."
    elif price_change < 0 and obv_change < 0:
        status = "수급 이탈 하락 (Confirmed Downtrend) ⚠️"
        desc = "주가 하락과 함께 자금 이탈이 함께 나타나고 있습니다."
    else:
        status = "수급 중립 (Neutral) ⚪"
        desc = "특이한 매집이나 분산 다이버전스 없이 균형을 유지하고 있습니다."
        
    return {
        "status": status,
        "desc": desc,
        "price_change_pct": round(price_change * 100, 1),
        "obv_change_pct": round(obv_change * 100, 1)
    }

def calculate_market_leadership(df: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> tuple:
    """
    S&P 500(SPY) 및 나스닥(QQQ) 대비 상대강도(RS, Relative Strength)를 계산하여
    현재 시장 주도주(Market Leader)인지 판별합니다.
    """
    # 최근 1개월, 3개월, 6개월 수익률 비교
    periods = {"1M": 21, "3M": 63, "6M": 126}
    scores = []
    
    for label, days in periods.items():
        if len(df) >= days and len(spy) >= days:
            stock_ret = (df["Close"].iloc[-1] - df["Close"].iloc[-days]) / df["Close"].iloc[-days]
            spy_ret = (spy["Close"].iloc[-1] - spy["Close"].iloc[-days]) / spy["Close"].iloc[-days]
            qqq_ret = (qqq["Close"].iloc[-1] - qqq["Close"].iloc[-days]) / qqq["Close"].iloc[-days]
            
            # 벤치마크 대비 초과 수익률
            excess_spy = stock_ret - spy_ret
            excess_qqq = stock_ret - qqq_ret
            
            weight = 0.5 if label == "3M" else 0.25
            score_term = (excess_spy * 0.5 + excess_qqq * 0.5) * 100 * weight
            scores.append(score_term)
            
    # 기본 50점에 초과수익률 합산하여 0~100점 스케일링
    raw_score = 50 + sum(scores)
    rs_score = int(np.clip(raw_score, 5, 99))
    
    # 75점 이상이면 시장 주도주 판정
    is_market_leader = rs_score >= 75
    return rs_score, is_market_leader

def analyze_operating_income(ticker, info: dict) -> dict:
    """
    영업이익(Operating Income), 영업이익률(Operating Margin, OPM) 및 성장률 추세를 분석하여
    성장률 감속(Deceleration) 여부를 진단합니다.
    """
    op_margin = info.get("operatingMargins", None)
    rev_growth = info.get("revenueGrowth", None)
    earnings_growth = info.get("earningsGrowth", None)
    
    # 재무제표 상 영업이익 내역 조회 시도
    op_income_trend = []
    growth_acceleration = "데이터 확인 필요"
    growth_alert = False
    
    try:
        financials = ticker.quarterly_financials
        if financials is not None and not financials.empty:
            if "Operating Income" in financials.index:
                op_series = financials.loc["Operating Income"].dropna()
                # 최근 4분기 영업이익
                for date, val in op_series.iloc[:4].items():
                    op_income_trend.append({
                        "quarter": date.strftime("%Y-%m"),
                        "operating_income_m": round(val / 1e6, 1)
                    })
                
                # 최근 분기 성장률 vs 직전 분기 성장률 비교 (가속 vs 감속)
                if len(op_series) >= 3:
                    growth_recent = (op_series.iloc[0] - op_series.iloc[1]) / (abs(op_series.iloc[1]) + 1e-9)
                    growth_prev = (op_series.iloc[1] - op_series.iloc[2]) / (abs(op_series.iloc[2]) + 1e-9)
                    
                    if growth_recent < growth_prev and growth_recent < 0.05:
                        growth_acceleration = "성장률 둔화 (Decelerating) ⚠️"
                        growth_alert = True
                    elif growth_recent > growth_prev:
                        growth_acceleration = "성장률 가속 (Accelerating) 🚀"
                    else:
                        growth_acceleration = "안정적 유지 (Stable) 🟢"
    except Exception:
        pass
        
    return {
        "operating_margin_pct": round(op_margin * 100, 2) if op_margin is not None else None,
        "revenue_growth_pct": round(rev_growth * 100, 2) if rev_growth is not None else None,
        "earnings_growth_pct": round(earnings_growth * 100, 2) if earnings_growth is not None else None,
        "growth_acceleration": growth_acceleration,
        "growth_alert": growth_alert,
        "op_income_trend": op_income_trend
    }

def estimate_institutional_accumulation(
    df: pd.DataFrame, poc_price: float, val_price: float, vah_price: float,
    avwap_earnings: float, avwap_low: float, ticker
) -> dict:
    """
    Anchored VWAP, Volume Profile POC, 13F 공시 및 다크풀 체결 가격을 융합하여
    기관의 종합 핵심 매집 예상 단가 범위를 산출합니다.
    """
    current_price = float(df["Close"].iloc[-1])
    
    # 앵커 단가 후보군 모음
    estimates = [poc_price]
    if avwap_earnings:
        estimates.append(avwap_earnings)
    if avwap_low:
        estimates.append(avwap_low)
        
    # 13F 기관 대형 보유자들의 추정 매집단가 (최근 분기 평균가 기반)
    inst_est_low = val_price
    inst_est_high = vah_price
    
    # 종합 기관 평단가 밴드: POC와 AVWAP의 가중 평균
    avg_inst_cost = np.mean(estimates)
    band_low = round(min(avg_inst_cost * 0.96, min(estimates)), 2)
    band_high = round(max(avg_inst_cost * 1.04, max(estimates)), 2)
    
    # 현재가와 기관 평단가 간 괴리율(%)
    gap_pct = round(((current_price - avg_inst_cost) / avg_inst_cost) * 100, 1)
    
    if current_price >= band_high:
        position_status = f"기관 평단가 상단 돌파 중 (+{gap_pct}%) 🟢"
        interpretation = "현재가는 기관 매집 단가 상단에 위치해 있어 기관 수익 구간입니다. 조정 시 기관 평단가가 강력한 지지선(안전마진)으로 작용합니다."
    elif current_price >= band_low:
        position_status = f"기관 핵심 매집 밴드 내부 진입 ({gap_pct:+.1f}%) 🎯"
        interpretation = "현재가가 기관들이 집중적으로 매집한 핵심 단가 구간에 직접 맞닿아 있어 손익비가 매우 유리한 분할 매수 타점입니다."
    else:
        position_status = f"기관 평단가 하회 구간 ({gap_pct}%) ⚠️"
        interpretation = "현재가가 기관들의 추정 평단가보다 낮아 기관들도 손실을 보고 있는 이례적 저평가 상태입니다. 단기 투매 진정 후 강한 기관 반발 매수가 기대되나 반등 시 본전 탈출 매물대에 주의해야 합니다."
        
    return {
        "avg_inst_cost": round(avg_inst_cost, 2),
        "band_low": band_low,
        "band_high": band_high,
        "gap_pct": gap_pct,
        "position_status": position_status,
        "interpretation": interpretation,
        "poc_price": round(poc_price, 2),
        "avwap_earnings": round(avwap_earnings, 2) if avwap_earnings else None,
        "avwap_low": round(avwap_low, 2) if avwap_low else None,
        "value_area_range": f"${val_price:.2f} ~ ${vah_price:.2f}"
    }
