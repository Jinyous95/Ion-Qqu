import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 자체 모듈 임포트
from data_engine import get_stock_data
from scenario_engine import calculate_5scenario_bands
from chart_builder import create_interactive_chart
from ai_advisor import generate_trading_plan

# 페이지 기본 설정 (와이드 모드, 다크 테마 감성)
st.set_page_config(
    page_title="US Stock 5-Band & Institutional Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일링 (모던 퀀트 터미널 룩)
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* 카드 컴포넌트 */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #8B949E;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F0F6FC;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-leader {
        background: rgba(46, 160, 67, 0.2);
        color: #3FB950;
        border: 1px solid #2EA043;
    }
    .badge-neutral {
        background: rgba(139, 148, 158, 0.2);
        color: #8B949E;
        border: 1px solid #6E7681;
    }
    .badge-alert {
        background: rgba(248, 81, 73, 0.2);
        color: #F85149;
        border: 1px solid #DA3633;
    }
    
    /* 📱 모바일 화면 최적화 (768px 이하 스마트폰 화면) */
    @media (max-width: 850px) {
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 0.8rem !important;
        }
        /* 좌우 분할 컬럼을 모바일에서는 100% 위아래 수직 배치로 강제 */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
        .metric-card {
            padding: 12px 14px !important;
            margin-bottom: 8px !important;
        }
        .metric-value {
            font-size: 1.35rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 분석 설정")
    
    st.subheader("🔍 미국 주식 티커 검색")
    quick_tickers = ["APP", "NVDA", "TSLA", "PLTR", "AAPL", "MSFT"]
    
    # 빠른 종목 선택 버튼
    selected_quick = st.pills("인기 종목 바로가기", quick_tickers, default=None)
    
    default_ticker = selected_quick if selected_quick else "APP"
    ticker_input = st.text_input("티커 심볼 입력 (예: APP, NVDA)", value=default_ticker).strip().upper()
    
    period_options = {"6개월": "6mo", "1년": "1y", "2년": "2y"}
    selected_period_label = st.radio("분석 기간", list(period_options.keys()), index=1, horizontal=True)
    period = period_options[selected_period_label]
    
    st.divider()
    
    st.subheader("📱 화면 레이아웃")
    layout_mode = st.radio(
        "배치 방식 선택",
        ["🖥️ 자동 반응형", "📱 위아래 세로 배치 (모바일 추천)", "💻 좌우 분할 (PC 추천)"],
        index=0,
        help="모바일에서는 '위아래 세로 배치'를 선택하시면 차트와 분석 리포트를 위아래로 넓고 시원하게 보실 수 있습니다."
    )
    
    st.divider()
    
    st.subheader("🤖 AI 연동 (선택사항)")
    api_key_input = st.text_input(
        "Google Gemini API Key", 
        type="password", 
        help="API 키를 입력하지 않아도 내장 퀀트 엔진으로 100% 무료 분석 리포트가 즉시 생성됩니다."
    )
    
    st.divider()
    
    with st.expander("📖 분석 모델 가이드 & 레퍼런스", expanded=False):
        st.markdown("""
        **1. 5단계 가치 밴드 (@irendebate 스타일)**
        - **Super Bear**: 시스템 위기 시 역사적 최저 PER/바닥
        - **Bear**: 보수적 멀티플 & 기관 매집 평단가 지지존
        - **Base**: 컨센서스 기준 정상 성장 적정가
        - **Bull**: 호실적 및 멀티플 확장 시 1차 목표가
        - **Super Bull**: 유동성 버블 및 리레이팅 오버슈팅
        
        **2. 기관 매집 평단가 추적 3대 축**
        - **Anchored VWAP**: 직전 실적일 및 바닥일 기점 평균가
        - **Volume Profile POC**: 최대 거래량 집중 매물대
        - **OBV / A-D**: 스텔스 매집 다이버전스 판별
        
        **3. 2대 필수 가중치**
        - **시장 주도주(RS)**: S&P/나스닥 대비 초과수익 탄력도
        - **영업이익(OPM)**: 성장률 가속 vs 감속(Deceleration) 경보
        """)

# 메인 타이틀 영역
st.markdown("### 🏛️ US Stock 5-Scenario Band & Institutional Terminal")
st.caption("포워드 실적 밴드 · 기관 수급 평단가(AVWAP & POC) · 시장 주도주 및 영업이익 퀄리티 진단")

if not ticker_input:
    st.info("좌측 사이드바에서 분석할 미국 주식 티커를 입력해주세요.")
    st.stop()

# 데이터 로딩 및 연산
try:
    with st.spinner(f"'{ticker_input}'의 실시간 주가, 재무제표, 기관 수급 데이터를 분석하는 중..."):
        stock_data = get_stock_data(ticker_input, period=period)
        scenario_data = calculate_5scenario_bands(stock_data)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 주요 상단 지표 카드 4분할
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

current_price = stock_data["current_price"]
prev_close = stock_data["prev_close"]
price_diff = current_price - prev_close
price_diff_pct = (price_diff / prev_close) * 100
inst = stock_data["inst_metrics"]
op = stock_data["op_metrics"]
rs = stock_data["rs_score"]
is_leader = stock_data["is_market_leader"]

with col_m1:
    diff_color = "#3FB950" if price_diff >= 0 else "#F85149"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{stock_data['company_name']} ({stock_data['ticker']})</div>
        <div class="metric-value">${current_price:.2f}</div>
        <div class="metric-sub" style="color: {diff_color}; font-weight: 600;">
            {price_diff:+.2f} ({price_diff_pct:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">5단계 밸류에이션 위치</div>
        <div class="metric-value" style="font-size: 1.25rem;">{scenario_data['current_band']}</div>
        <div class="metric-sub" style="color: #8B949E;">
            Base 적정가: ${scenario_data['bands']['Base']['low']} ~ ${scenario_data['bands']['Base']['high']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    gap = inst["gap_pct"]
    gap_col = "#3FB950" if gap >= 0 else "#F85149"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">기관 추정 평단가 (AVWAP+POC)</div>
        <div class="metric-value">${inst['avg_inst_cost']:.2f}</div>
        <div class="metric-sub" style="color: {gap_col}; font-weight: 600;">
            기관 대비 {gap:+.1f}% ({inst['band_low']} ~ {inst['band_high']})
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    badge_html = '<span class="badge badge-leader">주도주 (Leader)</span>' if is_leader else '<span class="badge badge-neutral">일반 종목</span>'
    alert_html = '<span class="badge badge-alert">⚠️ 성장률 감속</span>' if op.get("growth_alert") else '<span class="badge badge-leader">성장 정상</span>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">시장 주도주 & 영업이익 퀄리티</div>
        <div class="metric-value">RS {rs}점 {badge_html}</div>
        <div class="metric-sub" style="margin-top: 6px;">
            OPM: {op.get('operating_margin_pct', 'N/A')}% · {alert_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chart_section():
    st.markdown("#### 📈 기술적 수급 & 5단계 시나리오 밴드 차트")
    chart_h = 520 if "모바일" in layout_mode else 640
    chart_fig = create_interactive_chart(stock_data, scenario_data, height=chart_h)
    st.plotly_chart(chart_fig, width='stretch')
    
    st.markdown("##### 🎯 5단계 가치 밴드 상세 수치표")
    bands_dict = scenario_data["bands"]
    table_rows = []
    for band_name, b_info in bands_dict.items():
        is_here = "◀ 현재 위치" if band_name in scenario_data["current_band"] else ""
        table_rows.append({
            "시나리오": band_name,
            "가격 범위": f"${b_info['low']} ~ ${b_info['high']}",
            "타겟 P/E": f"{b_info['target_pe']}x",
            "상태 설명": b_info['desc'],
            "현재가 위치": is_here
        })
    df_bands = pd.DataFrame(table_rows)
    st.dataframe(df_bands, width='stretch', hide_index=True)

def render_report_section():
    st.markdown("#### 📑 AI 정밀 퀀트 분석 및 매매 계획서")
    with st.spinner("맞춤 매매 계획서를 도출하는 중..."):
        report_text = generate_trading_plan(stock_data, scenario_data, api_key=api_key_input)
    st.markdown(report_text)
    st.download_button(
        label="📥 분석 리포트 텍스트 다운로드 (.md)",
        data=report_text,
        file_name=f"{stock_data['ticker']}_valuation_report_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
        width='stretch'
    )

# 레이아웃 모드에 따른 분기
if layout_mode == "📱 위아래 세로 배치 (모바일 추천)":
    # 📱 모바일 세로 배치: 차트가 위에 크게 나오고, 리포트가 아래에 연속해서 나옴
    render_chart_section()
    st.divider()
    render_report_section()
else:
    # 💻 좌우 2분할 (PC 추천 및 자동 반응형)
    col_chart, col_report = st.columns([1.35, 1.0], gap="large")
    with col_chart:
        render_chart_section()
    with col_report:
        render_report_section()
