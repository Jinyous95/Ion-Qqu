import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_interactive_chart(stock_data: dict, scenario_data: dict, height: int = 620) -> go.Figure:
    """
    캔들스틱, 5단계 시나리오 밴드, 200일선, Anchored VWAP, 볼륨 프로파일(POC),
    기관 매집 구간이 통합된 전문가용 대시보드 인터랙티브 차트를 생성합니다.
    """
    df = stock_data["df"]
    ticker = stock_data["ticker"]
    company_name = stock_data["company_name"]
    inst = stock_data["inst_metrics"]
    bands = scenario_data["bands"]
    poc_price = inst["poc_price"]
    avwap_earnings = inst.get("avwap_earnings")
    avwap_low = inst.get("avwap_low")
    
    # 2개 서브플롯: 상단 캔들스틱 차트 (80%), 하단 볼륨 및 OBV (20%)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{company_name} ({ticker}) - 5단계 밸류에이션 밴드 & 기관 수급", "거래량 (Volume) & OBV"]
    )
    
    # 1. 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="주가 (OHLC)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # 2. 200일 이동평균선 (SMA 200)
    if "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                mode="lines",
                name="200일선 (SMA 200)",
                line=dict(color="#FFD700", width=2, dash="dash")
            ),
            row=1, col=1
        )
        
    # 3. Anchored VWAP (실적일 기준)
    if "AVWAP_Earnings" in df.columns and avwap_earnings:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AVWAP_Earnings"],
                mode="lines",
                name="Anchored VWAP (실적 앵커)",
                line=dict(color="#00E5FF", width=2.5)
            ),
            row=1, col=1
        )
        
    # 4. Anchored VWAP (52주 저점 앵커)
    if "AVWAP_Low" in df.columns and avwap_low:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AVWAP_Low"],
                mode="lines",
                name="Anchored VWAP (바닥 앵커)",
                line=dict(color="#76FF03", width=1.8, dash="dot")
            ),
            row=1, col=1
        )
        
    # 5. Volume Profile POC (Point of Control) 수평선
    fig.add_hline(
        y=poc_price,
        line_dash="dot",
        line_color="#E040FB",
        line_width=2.5,
        annotation_text=f"POC (최대 매물대): ${poc_price:.2f}",
        annotation_position="bottom right",
        annotation_font_color="#E040FB",
        row=1, col=1
    )
    
    # 6. 기관 핵심 매집 밴드 음영 영역 (POC와 AVWAP 수렴 구간)
    fig.add_hrect(
        y0=inst["band_low"],
        y1=inst["band_high"],
        fillcolor="rgba(156, 39, 176, 0.12)",
        line_width=1,
        line_color="rgba(156, 39, 176, 0.4)",
        annotation_text="기관 집중 매집 구간",
        annotation_position="left",
        row=1, col=1
    )
    
    # 7. 5단계 시나리오 밴드 가로 영역 (오른쪽 라벨 포함)
    band_colors = {
        "Super Bull": ("rgba(76, 175, 80, 0.12)", "#4CAF50"),
        "Bull": ("rgba(33, 150, 243, 0.12)", "#2196F3"),
        "Base": ("rgba(158, 158, 158, 0.12)", "#9E9E9E"),
        "Bear": ("rgba(255, 152, 0, 0.14)", "#FF9800"),
        "Super Bear": ("rgba(244, 67, 54, 0.14)", "#F44336")
    }
    
    for name, (fill_col, line_col) in band_colors.items():
        b = bands[name]
        fig.add_hrect(
            y0=b["low"],
            y1=b["high"],
            fillcolor=fill_col,
            line_width=1,
            line_color=line_col,
            row=1, col=1
        )
        # 밴드 대표 라인
        fig.add_hline(
            y=b["high"],
            line_dash="dash",
            line_color=line_col,
            line_width=1,
            annotation_text=f"[{name}] ${b['low']}~${b['high']}",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=line_col,
            row=1, col=1
        )
        
    # 8. 하단 서브플롯: 거래량 막대
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 레이아웃 스타일링 (고급 다크모드 금융 터미널 스타일)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(family="Pretendard, -apple-system, sans-serif")
    )
    
    fig.update_yaxes(title_text="주가 ($)", row=1, col=1, gridcolor="#21262D")
    fig.update_yaxes(title_text="거래량", row=2, col=1, gridcolor="#21262D")
    fig.update_xaxes(gridcolor="#21262D")
    
    return fig
