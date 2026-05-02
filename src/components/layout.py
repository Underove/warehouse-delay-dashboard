"""Dash 레이아웃. 운영자가 한눈에 위험 상황을 파악할 수 있게 정보 위계 강화."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# tooltip 박스를 dash가 아예 그리지 못하게 — readout이 옆에 있어 중복
HIDE_TOOLTIP = {
    "always_visible": False,
    "placement": "bottom",
    "style": {"display": "none", "visibility": "hidden", "opacity": 0},
}

SIGNAL_LABEL = {
    "order_inflow_15m": "주문 유입 (15분)",
    "congestion_score": "통로 혼잡도",
    "robot_active": "AMR 가동",
    "pack_utilization": "패킹 가동률",
}


def build_layout(
    layout_ids: list[str],
    default_layout: str,
    max_ts: int,
    signal_ranges: dict[str, tuple[float, float]],
) -> html.Div:
    return html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.Div(
                        className="app-header__brand",
                        children=[
                            html.H1("Warehouse Delay Dashboard"),
                            html.Span(
                                "AMR 기반 스마트 창고 30분 후 출고 지연 예측 · 예측 → 진단 → 시뮬레이션 한 화면에",
                                className="app-header__subtitle",
                            ),
                        ],
                    ),
                    html.Div(
                        id="status-pill",
                        className="status-pill status-pill--normal",
                        children="LIVE · MOCK",
                    ),
                ],
            ),
            html.Section(
                className="intro-strip",
                children=[
                    html.Div(
                        className="workflow",
                        children=[
                            _workflow_step("1", "Layout 선택", "분석할 창고 레이아웃 고르기"),
                            _workflow_step("2", "시점 이동", "15분 간격, 6시간 시뮬레이션 추적"),
                            _workflow_step("3", "위험 진단", "예측·원인·패턴 한눈에 확인"),
                            _workflow_step("4", "What-If (가상)", "신호를 바꿔 가상 시뮬레이션"),
                        ],
                    ),
                    html.Div(
                        className="risk-legend",
                        children=[
                            html.Span("위험도 기준", className="risk-legend__label"),
                            _risk_chip("정상", "< 15분", "normal"),
                            _risk_chip("경고", "15~22분", "warning"),
                            _risk_chip("임계", "≥ 22분", "critical"),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="controls",
                children=[
                    html.Div(
                        className="controls__group controls__group--stacked",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Layout (창고 레이아웃)"),
                                    dbc.Select(
                                        id="layout-picker",
                                        options=[{"label": lid, "value": lid} for lid in layout_ids],
                                        value=default_layout,
                                        className="dark-select",
                                    ),
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.Label("Scenario (시뮬레이션 #)"),
                                    dbc.Select(
                                        id="scenario-picker",
                                        options=[],
                                        value=None,
                                        className="dark-select",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="controls__group controls__group--wide",
                        children=[
                            html.Div(
                                className="controls__slider-header",
                                children=[
                                    html.Label(
                                        "시점 (15분 간격 · 한 시나리오 = 6시간 시뮬레이션)"
                                    ),
                                    html.Div(
                                        className="play-control",
                                        children=[
                                            html.Button(
                                                "▶ 재생",
                                                id="play-btn",
                                                className="play-btn",
                                                n_clicks=0,
                                            ),
                                            dbc.Select(
                                                id="play-speed",
                                                options=[
                                                    {"label": "0.5×", "value": "0.5"},
                                                    {"label": "1×", "value": "1"},
                                                    {"label": "2×", "value": "2"},
                                                    {"label": "4×", "value": "4"},
                                                ],
                                                value="1",
                                                className="dark-select play-speed",
                                            ),
                                            html.Span(id="ts-readout", className="ts-readout"),
                                        ],
                                    ),
                                    dcc.Interval(
                                        id="play-tick",
                                        interval=1000,
                                        disabled=True,
                                        n_intervals=0,
                                    ),
                                ],
                            ),
                            dcc.Input(
                                id="ts-slider",
                                type="range",
                                min=0,
                                max=max_ts - 1,
                                step=1,
                                value=0,
                                className="dark-range",
                            ),
                            html.Div(
                                className="time-bounds",
                                children=[
                                    html.Span("0분 (시뮬레이션 시작)", className="time-bounds__label"),
                                    html.Span("360분 (6시간 후)", className="time-bounds__label"),
                                ],
                            ),
                            html.Div(
                                className="sample-dots-wrap",
                                children=[
                                    html.Span(
                                        "각 칸 = 한 시점의 위험도 · 클릭으로 점프",
                                        className="sample-dots-hint",
                                    ),
                                    html.Div(id="timeline-dots", className="timeline-dots"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                id="layout-meta",
                className="layout-meta",
            ),
            html.Section(
                className="pipeline-section",
                children=[
                    _section_head(
                        "1",
                        "스마트 창고 물류 흐름",
                        "주문 유입부터 출고까지 6단계 · 단계 색깔 = 현재 위험 등급",
                        help_text=(
                            "AMR 기반 창고 운영 흐름을 6단계로 표현한 다이어그램. "
                            "각 카드는 그 단계의 핵심 신호(예: 충전·배차→저배터리 비율)와 "
                            "train 평균 대비 편차를 보여줍니다. "
                            "빨간 카드 = 임계 단계, 주황 = 경고."
                        ),
                    ),
                    html.Div(id="pipeline-stages", className="pipeline-stages"),
                ],
            ),
            html.Section(
                className="hero-section",
                children=[
                    _section_head(
                        "2",
                        "현재 예측",
                        "선택한 시점의 30분 후 평균 출고 지연 — 가운데 게이지로 위험도 한눈에",
                    ),
                    html.Div(
                        className="hero-row",
                        children=[
                            html.Div(id="hero-card", className="hero-card"),
                            html.Div(
                                className="gauge-card",
                                children=dcc.Loading(
                                    type="dot", color="#818cf8",
                                    children=dcc.Graph(
                                        id="gauge-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "260px"},
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="trend-section",
                children=[
                    _section_head(
                        "3",
                        "시점별 예측 추이 + 알림",
                        "6시간 시뮬레이션 동안 지연 변화 · 임계 진입/회복 자동 표시 · 흰 테두리 점 = 현재 시점",
                    ),
                    html.Div(id="alerts-strip", className="alerts-strip"),
                    dcc.Loading(
                        type="dot", color="#818cf8",
                        children=dcc.Graph(
                            id="trend-chart",
                            config={"displayModeBar": False},
                            style={"height": "340px"},
                        ),
                    ),
                ],
            ),
            html.Section(
                className="diagnose-wrap",
                children=[
                    _section_head(
                        "4",
                        "원인 분해와 현재 신호",
                        "예측을 만든 신호별 기여(분) + 현재 시점 핵심 신호값 · 미니 차트 = 6시간 추이",
                    ),
                    html.Div(
                        className="bottom-grid",
                        children=[
                            html.Div(
                                className="bottom-grid__panel",
                                children=[
                                    html.Div(
                                        className="panel-subhead",
                                        children=[
                                            html.Span("기여도 (분)", className="panel-subhead__title"),
                                            html.Span(
                                                "+ 면 지연 증가, − 면 감소 · Mock (모델 오면 SHAP)",
                                                className="panel-subhead__hint",
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="contrib-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "300px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="bottom-grid__panel",
                                children=[
                                    html.Div(
                                        className="panel-subhead",
                                        children=[
                                            html.Span("현재 시점 신호값", className="panel-subhead__title"),
                                            html.Span(
                                                "값 + train 평균 대비 % · 미니 차트는 6시간 추이",
                                                className="panel-subhead__hint",
                                            ),
                                        ],
                                    ),
                                    html.Div(id="signal-grid", className="signal-grid"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="diagnostic-section",
                children=[
                    _section_head(
                        "5",
                        "운영 패턴 진단",
                        "보라가 회색(평균) 밖으로 튀어나간 축이 위험 신호 · 빨간 꼭짓점은 평균보다 +25↑",
                        help_text=(
                            "Radar 차트: 핵심 신호 6개를 축으로 두고, "
                            "현재 시점(보라)을 train 전체 평균(회색 점선)과 겹쳐 그림. "
                            "축은 train의 1~99% 분위로 0~100 정규화. "
                            "보라가 회색을 크게 벗어나면 그 신호가 정상에서 멀어진 상태."
                        ),
                    ),
                    html.Div(
                        className="diagnostic-grid",
                        children=[
                            html.Div(
                                className="diagnostic-grid__panel",
                                children=dcc.Graph(
                                    id="schematic-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "440px"},
                                ),
                            ),
                            html.Div(
                                className="diagnostic-grid__panel diagnostic-grid__panel--list",
                                children=[
                                    html.Div(
                                        className="diagnostic-list-header",
                                        children=[
                                            html.Span("위험 신호 Top 3", className="diagnostic-list-header__title"),
                                            html.Span("z-score · 권장 액션", className="diagnostic-list-header__hint"),
                                        ],
                                    ),
                                    html.Div(id="bottleneck-list", className="bottleneck-list"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="whatif-section",
                children=[
                    _section_head(
                        "6",
                        "What-If 시뮬레이션",
                        "왼쪽 슬라이더로 신호를 조정하면 오른쪽에 즉시 재예측 결과가 나타납니다",
                        help_text=(
                            "운영 의사결정 시뮬레이터: 4개 핵심 신호를 슬라이더로 바꾼 가상 상황의 "
                            "예측을 현재와 비교. 위험도 전환(경고→임계 등)도 자동 표시. "
                            "현재 시점으로 리셋 버튼으로 baseline 복귀."
                        ),
                    ),
                    html.Div(
                        className="whatif-grid",
                        children=[
                            html.Div(
                                className="whatif-controls",
                                children=[
                                    *[
                                        _whatif_slider(col, signal_ranges.get(col, (0.0, 1.0)))
                                        for col in SIGNAL_LABEL
                                    ],
                                    html.Button(
                                        "현재 시점으로 리셋",
                                        id="whatif-reset",
                                        className="whatif-reset-btn",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="whatif-result",
                                children=[
                                    html.Div(id="whatif-comparison", className="whatif-comparison"),
                                    dcc.Graph(
                                        id="whatif-contrib-delta",
                                        config={"displayModeBar": False},
                                        style={"height": "260px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _workflow_step(num: str, title: str, desc: str) -> html.Div:
    return html.Div(
        className="workflow__step",
        children=[
            html.Span(num, className="workflow__num"),
            html.Div(
                className="workflow__body",
                children=[
                    html.Span(title, className="workflow__title"),
                    html.Span(desc, className="workflow__desc"),
                ],
            ),
        ],
    )


def _risk_chip(label: str, range_text: str, kind: str) -> html.Div:
    return html.Div(
        className=f"risk-chip risk-chip--{kind}",
        children=[
            html.Span(className=f"risk-chip__dot risk-chip__dot--{kind}"),
            html.Span(label, className="risk-chip__label"),
            html.Span(range_text, className="risk-chip__range"),
        ],
    )


def _section_head(num: str, title: str, hint: str, help_text: str | None = None) -> html.Div:
    children = [
        html.Span(num, className="section-header__num"),
        html.H2(title, className="section-header__title"),
        html.Span(hint, className="section-header__hint"),
    ]
    if help_text:
        children.append(html.Span(
            "ⓘ",
            className="section-header__help",
            title=help_text,
        ))
    return html.Div(
        className="section-header section-header--numbered",
        children=children,
    )


def _whatif_slider(col: str, rng: tuple[float, float]) -> html.Div:
    lo, hi = rng
    span = max(hi - lo, 1e-6)
    return html.Div(
        className="whatif-slider",
        children=[
            html.Div(
                className="whatif-slider__head",
                children=[
                    html.Span(SIGNAL_LABEL[col], className="whatif-slider__title"),
                    html.Span(id=f"whatif-{col}-readout", className="whatif-slider__readout"),
                ],
            ),
            dcc.Input(
                id=f"whatif-{col}-slider",
                type="range",
                min=lo,
                max=hi,
                step=span / 100,
                value=lo,
                className="dark-range",
            ),
        ],
    )
