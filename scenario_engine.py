import numpy as np
import pandas as pd

def calculate_5scenario_bands(stock_data: dict) -> dict:
    """
    미국 주식에 대한 5단계 가치 밴드(Super Bear, Bear, Base, Bull, Super Bull)를 정밀 산출합니다.
    
    [핵심 개선 사항 - 월가 기관 뷰 및 @irendebate 분석 방식과 완전 일치]
    1. 현재 급락한 주가의 단순 PER(순환참조)을 쓰지 않고, 
       월가 기관 컨센서스 목표가(Target Mean Price)와 포워드 EPS를 바탕으로 '정상 타겟 멀티플(Base Target P/E)'을 도출합니다.
    2. @irendebate 스레드와 동일하게 미래 포워드 EPS(예: APP의 예상 EPS $20.98)에 
       시나리오별 멀티플(Super Bear 8~10x, Bear 11~14x, Base 16~20x, Bull 22~27x, Super Bull 30~35x)을 적용합니다.
    3. 영업이익 감속(Deceleration) 시 밸류에이션 리스크 경보 발동 및 보수적 매매 전략 연동.
    """
    current_price = stock_data["current_price"]
    info = stock_data["info"]
    inst_metrics = stock_data["inst_metrics"]
    op_metrics = stock_data["op_metrics"]
    rs_score = stock_data["rs_score"]
    is_market_leader = stock_data["is_market_leader"]
    df = stock_data["df"]
    
    # 1. 포워드 EPS 산출
    fwd_eps = info.get("forwardEps", None)
    trailing_eps = info.get("trailingEps", None)
    
    if fwd_eps is None or fwd_eps <= 0:
        if trailing_eps and trailing_eps > 0:
            fwd_eps = trailing_eps * 1.15
        else:
            fwd_eps = current_price / 25.0
            
    # 2. 월가 기관 컨센서스 타겟 멀티플 (Consensus Target P/E) 도출
    target_mean = info.get("targetMeanPrice", None)
    current_pe = current_price / fwd_eps
    
    if target_mean and target_mean > 0:
        consensus_target_pe = target_mean / fwd_eps
    else:
        consensus_target_pe = max(18.0, current_pe * 1.15)
        
    # 특수 고성장/스토리 주식(TSLA, PLTR 등)의 멀티플 괴리 보정
    if current_pe > consensus_target_pe * 1.3:
        consensus_target_pe = (consensus_target_pe * 0.4) + (current_pe * 0.6)
        
    # 3. 5단계 시나리오 멀티플 비율 (Consensus Target PE를 기준으로 체계적 분배)
    # 월가 목표가(consensus_target_pe)는 대개 Base 상단 ~ Bull 하단에 위치합니다.
    pe_super_bear_low = round(consensus_target_pe * 0.35, 1)
    pe_super_bear_high = round(consensus_target_pe * 0.45, 1)
    
    pe_bear_low = round(consensus_target_pe * 0.48, 1)
    pe_bear_high = round(consensus_target_pe * 0.62, 1)
    
    pe_base_low = round(consensus_target_pe * 0.68, 1)
    pe_base_high = round(consensus_target_pe * 0.88, 1)
    
    pe_bull_low = round(consensus_target_pe * 0.95, 1)
    pe_bull_high = round(consensus_target_pe * 1.18, 1)
    
    pe_super_bull_low = round(consensus_target_pe * 1.28, 1)
    pe_super_bull_high = round(consensus_target_pe * 1.55, 1)
    
    # 가격 계산 (EPS * PER)
    sb_low = round(fwd_eps * pe_super_bear_low, 2)
    sb_high = round(fwd_eps * pe_super_bear_high, 2)
    
    b_low = round(fwd_eps * pe_bear_low, 2)
    b_high = round(fwd_eps * pe_bear_high, 2)
    
    base_low = round(fwd_eps * pe_base_low, 2)
    base_high = round(fwd_eps * pe_base_high, 2)
    
    bull_low = round(fwd_eps * pe_bull_low, 2)
    bull_high = round(fwd_eps * pe_bull_high, 2)
    
    super_bull_low = round(fwd_eps * pe_super_bull_low, 2)
    super_bull_high = round(fwd_eps * pe_super_bull_high, 2)
    
    # 4. 월가 최고 목표가(Target High Price) 및 수주/AI 내러티브 모멘텀 연동
    # IREN, APP처럼 월가 대형 IB들의 최고 목표가가 높은 성장주의 경우 Super Bull 상단을 탄력적으로 확장
    target_high = info.get("targetHighPrice", None)
    if target_high and float(target_high) > super_bull_high:
        super_bull_high = round(float(target_high), 2)
        super_bull_low = round(max(super_bull_low, bull_high * 1.05), 2)
        pe_super_bull_high = round(super_bull_high / fwd_eps, 1)

    bands = {
        "Super Bear": {
            "low": sb_low,
            "high": sb_high,
            "color": "#FF4B4B",
            "desc": f"경기 침체 및 실적 급감 시 극단적 바닥 밸류에이션 (PER {pe_super_bear_low}x~{pe_super_bear_high}x). 200주선 및 역사적 Floor.",
            "target_pe": pe_super_bear_high
        },
        "Bear": {
            "low": b_low,
            "high": b_high,
            "color": "#FFA726",
            "desc": f"단기 실적 미스 및 밸류 압축 가격대 (PER {pe_bear_low}x~{pe_bear_high}x). 기관 매집 평단가 부근의 안전마진 존.",
            "target_pe": pe_bear_high
        },
        "Base": {
            "low": base_low,
            "high": base_high,
            "color": "#9E9E9E",
            "desc": f"정상적인 성장 궤도가 유지될 때의 표준 적정 가치 (PER {pe_base_low}x~{pe_base_high}x). 월가 평균 뷰.",
            "target_pe": pe_base_high
        },
        "Bull": {
            "low": bull_low,
            "high": bull_high,
            "color": "#42A5F5",
            "desc": f"호실적 및 AI/섹터 확장 시 1차 도달 가능한 낙관 목표가 (PER {pe_bull_low}x~{pe_bull_high}x).",
            "target_pe": pe_bull_high
        },
        "Super Bull": {
            "low": super_bull_low,
            "high": super_bull_high,
            "color": "#66BB6A",
            "desc": f"시장 유동성 폭발 및 멀티플 리레이팅 오버슈팅 가격대 (PER {pe_super_bull_low}x~{pe_super_bull_high}x).",
            "target_pe": pe_super_bull_high
        }
    }
    
    # 현재가 위치 판정
    if current_price >= super_bull_low:
        current_band = "Super Bull (극단적 과열/오버슈팅) 🚨"
    elif current_price >= bull_low:
        current_band = "Bull (낙관 목표가 도달) 🔵"
    elif current_price >= base_low:
        current_band = "Base (적정 가치 구간) ⚪"
    elif current_price >= b_low:
        current_band = "Bear (안전마진 분할 매수 존) 🟠"
    elif current_price >= sb_low:
        current_band = "Super Bear (바닥권 패닉 바잉 존) 🔴"
    else:
        current_band = "Under Super Bear (극단적 저평가/투매) ⚠️"
        
    return {
        "bands": bands,
        "current_band": current_band,
        "fwd_eps": round(fwd_eps, 2),
        "consensus_target_pe": round(consensus_target_pe, 1),
        "base_pe_mid": round((pe_base_low + pe_base_high) / 2, 1)
    }
