import os
from google import genai
from google.genai import types

def generate_trading_plan(stock_data: dict, scenario_data: dict, api_key: str = None) -> str:
    """
    사용자의 투자 철학(@irendebate 스타일: 포워드 실적 밴드, 영업이익 감속 경계, 주도주 여부, 기관 매집 평단가)을
    적용하여 실전 맞춤 매매 계획서를 작성합니다.
    Gemini API Key가 있으면 LLM 추론을 사용하고, 없으면 정밀 규칙 기반 엔진으로 즉시 생성합니다.
    """
    ticker = stock_data["ticker"]
    company_name = stock_data["company_name"]
    current_price = stock_data["current_price"]
    prev_close = stock_data["prev_close"]
    pct_change = round(((current_price - prev_close) / prev_close) * 100, 2)
    
    inst = stock_data["inst_metrics"]
    op = stock_data["op_metrics"]
    rs_score = stock_data["rs_score"]
    is_leader = stock_data["is_market_leader"]
    obv = stock_data["obv_signal"]
    
    bands = scenario_data["bands"]
    current_band = scenario_data["current_band"]
    
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
당신은 월가의 탑티어 퀀트 헤지펀드 매니저이자, 철저하게 포워드 실적과 기관 수급, 영업이익 가속/감속을 기반으로 주가를 분석하는 전문 투자 분석가입니다.
(레퍼런스: Threads의 @irendebate 스타일 - 감정이 배제된 포워드 가치 밴드 역산, '싸다고 무조건 좋은 게 아니다'라는 성장률 둔화에 대한 냉철한 경계, 기관의 Anchored VWAP 및 POC 평단가 추적)

아래 분석 대상 종목의 실시간 정량 데이터를 바탕으로, 투자자에게 직관적이고 실행 가능한 [맞춤 시황 진단 및 실전 매매 계획서]를 작성해주세요.

[분석 종목 데이터]
- 종목: {company_name} ({ticker})
- 현재가: ${current_price:.2f} (전일 대비 {pct_change:+.2f}%)
- 현재 밴드 위치: {current_band}

[5단계 가치 밴드]
- Super Bear: ${bands['Super Bear']['low']} ~ ${bands['Super Bear']['high']} (Target PER: {bands['Super Bear']['target_pe']}x)
- Bear: ${bands['Bear']['low']} ~ ${bands['Bear']['high']} (Target PER: {bands['Bear']['target_pe']}x)
- Base: ${bands['Base']['low']} ~ ${bands['Base']['high']} (Target PER: {bands['Base']['target_pe']}x)
- Bull: ${bands['Bull']['low']} ~ ${bands['Bull']['high']} (Target PER: {bands['Bull']['target_pe']}x)
- Super Bull: ${bands['Super Bull']['low']} ~ ${bands['Super Bull']['high']} (Target PER: {bands['Super Bull']['target_pe']}x)

[기관 매집 수급 데이터]
- 종합 기관 추정 평단가: ${inst['avg_inst_cost']:.2f} (매집 밴드: ${inst['band_low']} ~ ${inst['band_high']})
- 현재가와 기관 평단가 괴리율: {inst['gap_pct']:+.1f}% ({inst['position_status']})
- Volume Profile POC (최대 거래량 단가): ${inst['poc_price']}
- Anchored VWAP (실적일 기준): ${inst['avwap_earnings']}
- OBV / A-D 수급 신호: {obv['status']} ({obv['desc']})

[시장 지위 & 영업이익 펀더멘털]
- 시장 주도주(RS) 점수: {rs_score}점 / 100점 (주도주 여부: {'예' if is_leader else '아니오'})
- 영업이익률(OPM): {op.get('operating_margin_pct', 'N/A')}%
- 영업이익 성장 궤적: {op['growth_acceleration']}
- 성장률 둔화 경보(Deceleration Alert): {'⚠️ 경보 발동 (디레이팅 주의)' if op['growth_alert'] else '정상 (성장 유지/가속)'}

[작성 가이드라인]
1. **총평 및 현재가 진단**: 현재가가 5단계 밴드 중 어디에 위치하며, 밸류에이션 매력도와 리스크는 무엇인가?
2. **기관 수급 및 매집 단가 분석**: 기관 평단가와 현재가의 관계(지지선 여부, 물림 여부)와 OBV 다이버전스 해석.
3. **영업이익 및 성장률 팩트체크**: "싸다고 무조건 사면 안 되는 이유(성장률 둔화 여부)"를 반영한 펀더멘털 평가.
4. **구체적 실전 매매 계획 (Action Plan)**:
   - 1차 분할 매수 가격대 및 비중(%)
   - 2차 추가 매수(불타기 or 물타기) 트리거 조건
   - 목표 익절가 (1차 Bull, 2차 Super Bull) 및 비중
   - 칼같은 손절 및 비중 축소 기준 (리스크 관리)
마크다운 형식으로 가독성 높고 세련되게 작성해주세요.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            # API 호출 실패 시 규칙 기반 엔진으로 자연스럽게 폴백
            pass

    # 규칙 기반 고품질 리포트 자동 생성 엔진 (오프라인/API키 없을 때 즉시 동작)
    return generate_rule_based_report(stock_data, scenario_data)

def generate_rule_based_report(stock_data: dict, scenario_data: dict) -> str:
    """오프라인 또는 API 키 미입력 시 작동하는 고정밀 알고리즘 분석 리포트"""
    ticker = stock_data["ticker"]
    company_name = stock_data["company_name"]
    current_price = stock_data["current_price"]
    inst = stock_data["inst_metrics"]
    op = stock_data["op_metrics"]
    rs_score = stock_data["rs_score"]
    is_leader = stock_data["is_market_leader"]
    obv = stock_data["obv_signal"]
    bands = scenario_data["bands"]
    current_band = scenario_data["current_band"]
    
    # 매매 플랜 전략 도출
    if "Super Bear" in current_band or "Under" in current_band:
        action_strategy = "패닉 투매 구간 적극 분할 매수 (High Risk / Maximum Reward)"
        buy_plan = f"• **1차 진입**: 현재가(${current_price:.2f}) 부근에서 포트폴리오의 30% 진입\n• **2차 진입**: ${bands['Super Bear']['low']} 지지 확인 시 30% 추가 매수 (총 60% 비중)"
        sell_plan = f"• **1차 목표가**: Base 구간 하단 (${bands['Base']['low']}) 도달 시 50% 분할 익절\n• **2차 목표가**: Bull 구간 (${bands['Bull']['low']}) 도달 시 잔여 물량 정리"
        stop_plan = f"• **손절 기준**: 실적 추정치 추가 하향과 함께 ${bands['Super Bear']['low'] * 0.93:.2f} 일봉 종가 이탈 시 비중 축소"
    elif "Bear" in current_band:
        action_strategy = "안전마진 확보 분할 매수 존 (기관 매집 평단가 지지 테스트)"
        buy_plan = f"• **1차 진입**: 기관 평단가(${inst['avg_inst_cost']:.2f}) 및 Bear 지지선 부근에서 25~35% 진입\n• **2차 진입**: 직전 실적일 AVWAP(${inst['avwap_earnings'] or current_price:.2f}) 돌파 안착 시 25% 불타기"
        sell_plan = f"• **1차 목표가**: Base 구간 상단 (${bands['Base']['high']})에서 30% 익절\n• **2차 목표가**: Bull 구간 (${bands['Bull']['low']} ~ ${bands['Bull']['high']})에서 40% 분할 익절"
        stop_plan = f"• **손절 기준**: 기관 매집 하단(${inst['band_low']}) 및 ${bands['Super Bear']['high']} 진입 시 50% 손절 대응"
    elif "Base" in current_band:
        action_strategy = "중립 및 관망 (Hold & Watch) — 추세 확인 후 진입"
        buy_plan = f"• **신규 진입**: 현재가는 적정가 구간이므로 무리한 추격 매수 자제. Bear 밴드(${bands['Bear']['high']})로 눌림목 발생 시 분할 매수 대기"
        sell_plan = f"• **기존 보유자**: Bull 밴드(${bands['Bull']['low']}) 진입 전까지 홀딩하며 추세 추종"
        stop_plan = f"• **리스크 관리**: 200일선 및 Bear 상단(${bands['Bear']['high']}) 이탈 시 비중 20% 축소"
    elif "Bull" in current_band:
        action_strategy = "수익 실현 및 트레일링 스탑 구간 (1차 익절 실행)"
        buy_plan = "• **신규 매수**: 밸류에이션 상단 영역이므로 신규 매수 금지"
        sell_plan = f"• **1차 익절**: 현재가(${current_price:.2f}) 부근에서 보유 물량의 40% 차익 실현\n• **2차 목표가**: Super Bull 하단 (${bands['Super Bull']['low']}) 도달 시 잔여 물량의 절반 추가 익절"
        stop_plan = f"• **트레일링 스탑**: 최근 20일선 또는 Base 상단(${bands['Base']['high']})을 이탈하지 않는 한 잔여 물량으로 랠리 향유"
    else:
        action_strategy = "극단적 과열 경보 — 전량 차익 실현 및 리스크 오프"
        buy_plan = "• **신규 매수**: 절대 금지 (FOMO 추격 매수 위험 최상위)"
        sell_plan = f"• **수익 확정**: 보유 물량의 70~100%를 현금화하여 이익 보존\n• **상단 목표**: ${bands['Super Bull']['high']} 오버슈팅 시 전량 매도"
        stop_plan = f"• **익절 마지노선**: ${bands['Bull']['high']} 이탈 시 전량 시장가 익절"

    deceleration_text = (
        "⚠️ **[주의] 최근 분기 영업이익 성장률 둔화(Deceleration) 신호 감지!**\n"
        "스노우플레이크/앱러빈 사례처럼 단순히 주가가 싸졌다고 매수하면 미래 기대치(PER)가 낮아지는 디레이팅 함정에 빠질 수 있습니다. 밸류에이션 밴드 상단을 보수적으로 제한하고 분할 매수 호흡을 길게 가져가야 합니다."
        if op.get("growth_alert") else
        "🟢 **영업이익 성장세 양호**: 본업의 이익 체력이 견고하여 밸류에이션 신뢰도가 높습니다."
    )

    report = f"""
### 📊 {company_name} ({ticker}) 종합 시황 및 매매 계획서

---

#### 1. 🎯 현재 밸류에이션 위치: **{current_band}**
- **현재가**: **${current_price:.2f}**
- **포워드 밸류에이션 진단**: 
  - 기본 가치(Base): **${bands['Base']['low']} ~ ${bands['Base']['high']}** (Target PER: {bands['Base']['target_pe']}x)
  - 하방 안전마진(Bear): **${bands['Bear']['low']} ~ ${bands['Bear']['high']}**
  - 상방 목표가(Bull): **${bands['Bull']['low']} ~ ${bands['Bull']['high']}**

---

#### 2. 🏛️ 기관 수급 & 매집 단가(Order Flow) 정밀 진단
- **종합 기관 추정 평단가**: **${inst['avg_inst_cost']:.2f}** (매집 밴드: `${inst['band_low']} ~ ${inst['band_high']}`)
- **현재가 대비 위치**: **{inst['position_status']}**
  - *{inst['interpretation']}*
- **핵심 수급 레벨**:
  - **Volume Profile POC (최대 매물대)**: **${inst['poc_price']}**
  - **Anchored VWAP (직전 실적일 앵커)**: **${inst['avwap_earnings'] if inst['avwap_earnings'] else 'N/A'}**
- **OBV / A-D Line 진단**: **{obv['status']}**
  - *{obv['desc']}*

---

#### 3. 🚀 시장 주도주 & 💼 영업이익 퀄리티 체크
- **시장 주도주(RS) 지수**: **{rs_score}점** ({'🏆 시장 주도주 대장주 그룹에 속함' if is_leader else '일반/중립 종목'})
- **영업이익률(OPM)**: **{op.get('operating_margin_pct', 'N/A')}%**
- **영업이익 성장 추세**: **{op['growth_acceleration']}**
- {deceleration_text}

---

#### 4. 📋 실전 매매 액션 플랜 (Action Plan)
**전략 기조: [{action_strategy}]**

##### 🛒 매수 전략
{buy_plan}

##### 💰 매도 및 익절 전략
{sell_plan}

##### 🛡️ 리스크 관리 (손절 기준)
{stop_plan}
"""
    return report.strip()
