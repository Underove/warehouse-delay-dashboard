"""Dash 엔트리. Phase 1 시각화 + Phase 2 What-If Simulator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, html, no_update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ASSETS_DIR, RISK_THRESHOLDS, SEQ_LEN, TARGET
from src.components.layout import build_layout
from src.data_loader import load_layout, load_test, load_train
from src.inference import _classify_risk, load_predictor

RISK_COLOR = {
    "normal": "#10b981",
    "warning": "#f59e0b",
    "critical": "#ef4444",
}
RISK_LABEL = {"normal": "정상", "warning": "경고", "critical": "임계"}
SIGNAL_COLS = ("order_inflow_15m", "congestion_score", "robot_active", "pack_utilization")
SIGNAL_LABEL = {
    "order_inflow_15m": "주문 유입 (15m)",
    "congestion_score": "혼잡도",
    "robot_active": "로봇 가동",
    "pack_utilization": "패킹 가동률",
}
GAUGE_MAX = 40.0


def _prepare_test(test: pd.DataFrame) -> pd.DataFrame:
    """ID 기준 정렬(시간 순서) 후 scenario별 ts_rank 부여 (15분 간격, 25 timestep)."""
    if "scenario_id" not in test.columns:
        return test
    test = test.copy()
    if "ID" in test.columns:
        test = test.sort_values("ID").reset_index(drop=True)
    if "ts_rank" not in test.columns:
        test["ts_rank"] = test.groupby("scenario_id").cumcount()
    return test


PIPELINE_STAGES = [
    {
        "key": "inbound",
        "icon": "📥",
        "title": "주문 유입",
        "desc": "출고 주문이 시스템에 들어옴",
        "metric_col": "order_inflow_15m",
        "metric_label": "주문 (15분)",
        "risk_dir": +1,
    },
    {
        "key": "pick",
        "icon": "🤖",
        "title": "로봇 피킹",
        "desc": "AGV/AMR이 SKU를 가져옴",
        "metric_col": "robot_active",
        "metric_label": "가동 로봇",
        "risk_dir": +1,
    },
    {
        "key": "aisle",
        "icon": "🚥",
        "title": "통로 이동",
        "desc": "통로 혼잡 · 충돌 · 우회",
        "metric_col": "congestion_score",
        "metric_label": "혼잡도",
        "risk_dir": +1,
    },
    {
        "key": "pack",
        "icon": "📦",
        "title": "패킹",
        "desc": "포장 · 라벨링 · 검수",
        "metric_col": "pack_utilization",
        "metric_label": "패킹 가동률",
        "risk_dir": +1,
    },
    {
        "key": "outbound",
        "icon": "🚚",
        "title": "출고",
        "desc": "트럭 적재 · 도크 대기",
        "metric_col": "outbound_truck_wait_min",
        "metric_label": "도크 대기 (분)",
        "risk_dir": +1,
    },
]

LAYOUT_TYPE_KR = {
    "narrow": "좁은 통로형",
    "grid": "격자형",
    "hybrid": "혼합형",
    "hub_spoke": "허브-스포크형",
}


RADAR_COLS = (
    "order_inflow_15m",
    "congestion_score",
    "robot_active",
    "pack_utilization",
    "aisle_traffic_score",
    "path_optimization_score",
)
RADAR_LABEL = {
    "order_inflow_15m": "주문 유입",
    "congestion_score": "혼잡도",
    "robot_active": "로봇 가동",
    "pack_utilization": "패킹 가동률",
    "aisle_traffic_score": "통로 혼잡",
    "path_optimization_score": "경로 최적성",
}
# +1: 높을수록 위험, -1: 낮을수록 위험
RADAR_RISK_DIR = {
    "order_inflow_15m": +1,
    "congestion_score": +1,
    "robot_active": +1,
    "pack_utilization": +1,
    "aisle_traffic_score": +1,
    "path_optimization_score": -1,
}

LAYOUT_META_PREFERRED = [
    ("layout_type", "TYPE (레이아웃 유형)"),
    ("layout_class", "TYPE (레이아웃 유형)"),
    ("intersection_count", "INTERSECTIONS (교차점 수)"),
    ("aisle_width", "AISLE WIDTH (통로 폭)"),
    ("one_way_ratio", "ONE-WAY (일방통행 비율)"),
    ("bottleneck_intensity", "BOTTLENECK (병목 강도)"),
    ("total_zones", "ZONES (구역 수)"),
    ("pack_station_count", "PACK STATIONS (패킹대 수)"),
    ("staff_on_floor", "STAFF (작업자 수)"),
]


def main() -> None:
    train = load_train()
    test = _prepare_test(load_test())
    layout_info = load_layout()
    predictor = load_predictor(ref_df=train)

    signal_means = {c: float(train[c].mean(skipna=True)) for c in SIGNAL_COLS if c in train.columns}
    signal_ranges = {
        c: (float(train[c].quantile(0.01)), float(train[c].quantile(0.99)))
        for c in SIGNAL_COLS if c in train.columns
    }
    radar_norm = {}
    for c in RADAR_COLS:
        if c in train.columns:
            p1 = float(train[c].quantile(0.01))
            p99 = float(train[c].quantile(0.99))
            mean = float(train[c].mean(skipna=True))
            std = float(train[c].std(skipna=True))
            radar_norm[c] = {"p1": p1, "p99": p99, "mean": mean, "std": std or 1.0}

    # 파이프라인 단계별 train 통계 (mean, std)
    pipeline_stats = {}
    for stage in PIPELINE_STAGES:
        c = stage["metric_col"]
        if c in train.columns:
            m = float(train[c].mean(skipna=True))
            s = float(train[c].std(skipna=True))
            pipeline_stats[c] = {"mean": m, "std": s or 1.0}

    # 시간대 패턴 (Phase 5): 요일 × 시간대 평균 지연
    if "day_of_week" in train.columns and "shift_hour" in train.columns:
        pat = (
            train.dropna(subset=["day_of_week", "shift_hour"])
            .groupby(["day_of_week", "shift_hour"])[TARGET]
            .mean()
            .unstack()
            .sort_index()
            .reindex(columns=range(24))
        )
        pattern_pivot = pat
        pattern_overall_mean = float(train[TARGET].mean())
    else:
        pattern_pivot = None
        pattern_overall_mean = float(train[TARGET].mean())

    layout_meta_cols = [(c, label) for c, label in LAYOUT_META_PREFERRED if c in layout_info.columns]
    if not layout_meta_cols:
        # fallback: layout_info의 layout_id 외 처음 4개 컬럼
        rest = [c for c in layout_info.columns if c != "layout_id"][:4]
        layout_meta_cols = [(c, c.upper()) for c in rest]
    print(f"[boot] layout meta cols: {[c for c, _ in layout_meta_cols]}")

    layout_info_idx = (
        layout_info.set_index("layout_id") if "layout_id" in layout_info.columns else layout_info
    )

    layout_ids = sorted(test["layout_id"].astype(str).unique().tolist())
    default_layout = layout_ids[0]

    # scenario별 (seq, preds) 캐시 — 슬라이더 드래그 시 25 prediction 재계산 방지
    scenario_cache: dict[str, tuple[pd.DataFrame, list]] = {}

    def _scenario(layout_id_str: str) -> tuple[pd.DataFrame, list] | tuple[None, None]:
        if layout_id_str in scenario_cache:
            return scenario_cache[layout_id_str]
        scenario = test[test["layout_id"].astype(str) == layout_id_str]
        if scenario.empty:
            return None, None
        first_scenario_id = scenario["scenario_id"].iloc[0]
        seq = (
            scenario[scenario["scenario_id"] == first_scenario_id]
            .sort_values("ts_rank")
            .reset_index(drop=True)
        )
        preds = [predictor.predict_one(r) for _, r in seq.iterrows()]
        scenario_cache[layout_id_str] = (seq, preds)
        return seq, preds

    app = Dash(__name__, assets_folder=str(ASSETS_DIR))
    app.title = "Warehouse Delay Dashboard"
    # head에 inline <style> 강제 주입 — 브라우저 캐시/specificity 우회
    app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        /* dbc.Select 강제 다크 — native <select>이라 무조건 잡힌다 */
        #layout-picker.form-select,
        #layout-picker {
            background-color: #16223d !important;
            color: #e2e8f0 !important;
            border: 1px solid #1f2a44 !important;
            border-radius: 8px !important;
            padding: 9px 36px 9px 14px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>") !important;
            background-repeat: no-repeat !important;
            background-position: right 12px center !important;
            -webkit-appearance: none !important;
            appearance: none !important;
        }
        #layout-picker:hover { border-color: #818cf8 !important; }
        #layout-picker:focus {
            border-color: #818cf8 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
            outline: none !important;
        }
        #layout-picker option {
            background-color: #16223d !important;
            color: #e2e8f0 !important;
        }
        /* 슬라이더 tooltip — 아예 숨김. readout이 옆에 있어 중복 */
        .rc-slider-tooltip,
        [class*="rc-slider-tooltip"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""
    app.layout = build_layout(
        layout_ids=layout_ids,
        default_layout=default_layout,
        max_ts=SEQ_LEN,
        signal_ranges=signal_ranges,
    )

    def _current_row(layout_id: str, ts: int) -> tuple[pd.Series | None, list, list]:
        seq, preds = _scenario(str(layout_id))
        if seq is None:
            return None, [], []
        ts = max(0, min(ts, len(seq) - 1))
        return seq.iloc[ts], preds, list(seq.index)

    @app.callback(
        Output("hero-card", "children"),
        Output("gauge-chart", "figure"),
        Output("trend-chart", "figure"),
        Output("contrib-chart", "figure"),
        Output("signal-grid", "children"),
        Output("timeline-dots", "children"),
        Output("ts-readout", "children"),
        Output("status-pill", "className"),
        Output("layout-meta", "children"),
        Output("schematic-chart", "figure"),
        Output("bottleneck-list", "children"),
        Output("alerts-strip", "children"),
        Output("pattern-chart", "figure"),
        Output("pattern-side", "children"),
        Output("pipeline-stages", "children"),
        Input("layout-picker", "value"),
        Input("ts-slider", "value"),
    )
    def update_main(layout_id: str, ts):
        try:
            ts = int(ts) if ts is not None else 0
        except (TypeError, ValueError):
            ts = 0
        seq, preds = _scenario(str(layout_id))
        if seq is None:
            empty = go.Figure()
            return (
                html.Div("데이터 없음"), empty, empty, empty, [], [],
                f"0/{SEQ_LEN}", "status-pill status-pill--normal", [], empty, [], [],
                empty, [], [],
            )
        ts = max(0, min(ts, len(seq) - 1))
        values = [p.value for p in preds]
        risks = [p.risk for p in preds]
        current = preds[ts]
        current_row = seq.iloc[ts]
        layout_row = (
            layout_info_idx.loc[str(layout_id)]
            if str(layout_id) in layout_info_idx.index else None
        )
        if isinstance(layout_row, pd.DataFrame):
            layout_row = layout_row.iloc[0]
        schematic = _radar_figure(current_row, radar_norm)
        bottlenecks = _top_risk_signals(current_row, radar_norm)
        events_info = _detect_events(preds)
        cur_dow = current_row.get("day_of_week")
        cur_hour = current_row.get("shift_hour")
        cur_dow = int(cur_dow) if pd.notna(cur_dow) else None
        cur_hour = int(cur_hour) if pd.notna(cur_hour) else None
        pattern_fig = _pattern_figure(pattern_pivot, cur_dow, cur_hour)
        pattern_side_card = _pattern_side(
            pattern_pivot, pattern_overall_mean, cur_dow, cur_hour, current.value
        )
        pipeline = _pipeline_stages(current_row, pipeline_stats)

        layout_type = None
        if layout_row is not None and "layout_type" in layout_row.index:
            lt = layout_row.get("layout_type")
            layout_type = LAYOUT_TYPE_KR.get(str(lt), str(lt)) if pd.notna(lt) else None

        return (
            _hero(current, ts, len(seq), layout_type, current_row, radar_norm),
            _gauge_figure(current.value, current.risk),
            _trend_figure(values, risks, ts, events_info),
            _contrib_figure(current.contributions or {}),
            _signal_cards(current_row, signal_means, seq),
            _timeline_dots(risks, ts),
            f"{_ts_to_clock(ts)} · {ts + 1}/{len(seq)} 시점",
            f"status-pill status-pill--{current.risk}",
            _layout_meta(layout_info_idx, str(layout_id), layout_meta_cols),
            schematic,
            bottlenecks,
            _alerts_strip(events_info, ts, current),
            pattern_fig,
            pattern_side_card,
            pipeline,
        )

    # What-If: layout/ts/reset 트리거 → 4개 슬라이더를 현재 row 값으로 리셋
    @app.callback(
        [Output(f"whatif-{c}-slider", "value") for c in SIGNAL_COLS],
        Input("layout-picker", "value"),
        Input("ts-slider", "value"),
        Input("whatif-reset", "n_clicks"),
    )
    def reset_whatif_sliders(layout_id, ts, _n_clicks):
        try:
            ts = int(ts) if ts is not None else 0
        except (TypeError, ValueError):
            ts = 0
        row, _preds, _ = _current_row(layout_id, ts)
        if row is None:
            return [signal_ranges[c][0] for c in SIGNAL_COLS]
        out = []
        for c in SIGNAL_COLS:
            v = row.get(c)
            if v is None or pd.isna(v):
                out.append(float(signal_means.get(c, signal_ranges[c][0])))
            else:
                lo, hi = signal_ranges[c]
                out.append(float(min(max(float(v), lo), hi)))
        return out

    # What-If: 슬라이더 변경 → 비교 카드 + delta 차트 + readout
    whatif_inputs = [Input(f"whatif-{c}-slider", "value") for c in SIGNAL_COLS]

    @app.callback(
        Output("whatif-comparison", "children"),
        Output("whatif-contrib-delta", "figure"),
        [Output(f"whatif-{c}-readout", "children") for c in SIGNAL_COLS],
        Input("layout-picker", "value"),
        Input("ts-slider", "value"),
        *whatif_inputs,
    )
    def update_whatif(layout_id, ts, *slider_vals):
        try:
            ts = int(ts) if ts is not None else 0
        except (TypeError, ValueError):
            ts = 0
        row, _preds, _ = _current_row(layout_id, ts)
        if row is None:
            empty = go.Figure()
            return html.Div("데이터 없음"), empty, *(["—"] * len(SIGNAL_COLS))

        def _to_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        slider_vals = [_to_float(v) for v in slider_vals]

        base_pred = predictor.predict_one(row)

        overridden = row.copy()
        for c, v in zip(SIGNAL_COLS, slider_vals):
            if v is not None:
                overridden[c] = float(v)
        new_pred = predictor.predict_one(overridden)

        readouts = []
        for c, v in zip(SIGNAL_COLS, slider_vals):
            current_val = row.get(c)
            if v is None:
                readouts.append("—")
                continue
            v = float(v)
            if current_val is None or pd.isna(current_val):
                readouts.append(f"{v:.2f}")
            else:
                cur = float(current_val)
                pct = (v - cur) / abs(cur) * 100 if cur != 0 else 0
                if abs(v - cur) < 1e-6:
                    readouts.append(f"{v:.2f}  · 기본")
                else:
                    sign = "+" if pct >= 0 else ""
                    readouts.append(f"{v:.2f}  ({sign}{pct:.0f}%)")

        comparison = _whatif_comparison(base_pred, new_pred)
        delta_fig = _whatif_delta_figure(base_pred.contributions or {}, new_pred.contributions or {})
        return comparison, delta_fig, *readouts

    app.run(debug=True, port=8050)


# -- 차트/카드 빌더 ---------------------------------------------------------

def _hero(pred, ts: int, total: int, layout_type: str | None = None,
          current_row: pd.Series | None = None,
          radar_norm: dict | None = None) -> html.Div:
    color = RISK_COLOR[pred.risk]
    headroom = RISK_THRESHOLDS["critical"] - pred.value
    headroom_label = "여유" if headroom >= 0 else "초과"
    headroom_color = "#10b981" if headroom >= 0 else "#ef4444"
    narrative = _narrative_text(pred, layout_type, current_row, radar_norm)
    return html.Div(
        children=[
            html.Div(
                className="hero-card__narrative",
                style={"borderLeftColor": color},
                children=narrative,
            ),
            html.Div(
                className="hero-card__top",
                children=[
                    html.Span("예측 지연", className="hero-card__label"),
                    html.Span(
                        RISK_LABEL[pred.risk],
                        className="hero-card__risk",
                        style={"backgroundColor": color},
                    ),
                ],
            ),
            html.Div(
                className="hero-card__value",
                children=[
                    html.Span(f"{pred.value:.1f}", style={"color": color}),
                    html.Span("분", className="hero-card__unit"),
                ],
            ),
            html.Div(
                className="hero-card__meta",
                children=[
                    html.Div(
                        className="hero-card__meta-item",
                        children=[
                            html.Span("임계까지", className="hero-card__meta-label"),
                            html.Span(
                                f"{headroom:+.1f}분 {headroom_label}",
                                className="hero-card__meta-value",
                                style={"color": headroom_color},
                            ),
                        ],
                    ),
                    html.Div(
                        className="hero-card__meta-item",
                        children=[
                            html.Span("시점", className="hero-card__meta-label"),
                            html.Span(
                                f"{_ts_to_clock(ts)} ({ts * 15}분 경과)",
                                className="hero-card__meta-value",
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


def _narrative_text(pred, layout_type: str | None,
                    current_row: pd.Series | None,
                    radar_norm: dict | None) -> str:
    risk_text = {"normal": "정상", "warning": "경고", "critical": "임계"}[pred.risk]
    base = (
        f"{layout_type + ' 창고' if layout_type else '이 창고'} · "
        f"30분 후 {pred.value:.1f}분 지연 예상 ({risk_text})"
    )
    if current_row is None or radar_norm is None:
        return base + "."
    # 위험 방향 z-score top
    deviations = []
    for c in RADAR_COLS:
        if c not in radar_norm:
            continue
        v = current_row.get(c)
        if v is None or pd.isna(v):
            continue
        z = (float(v) - radar_norm[c]["mean"]) / radar_norm[c]["std"]
        deviations.append((c, z * RADAR_RISK_DIR.get(c, 1)))
    if not deviations:
        return base + "."
    deviations.sort(key=lambda x: x[1], reverse=True)
    top_col, top_z = deviations[0]
    if top_z >= 0.5:
        return f"{base}. 가장 큰 원인: {RADAR_LABEL[top_col]}이(가) 평균보다 +{top_z:.1f}σ 높음."
    return f"{base}. 모든 신호가 평균 범위 내 — 정상 운영 중."


def _pipeline_stages(row: pd.Series, stats: dict) -> list:
    cards = []
    for i, stage in enumerate(PIPELINE_STAGES):
        col = stage["metric_col"]
        v = row.get(col) if col in row else None
        if v is None or pd.isna(v) or col not in stats:
            risk = "normal"
            value_text = "—"
            delta_text = "데이터 없음"
        else:
            v = float(v)
            s = stats[col]
            z = (v - s["mean"]) / s["std"]
            signed_z = z * stage.get("risk_dir", 1)
            risk = (
                "critical" if signed_z >= 1.5 else
                "warning" if signed_z >= 0.5 else
                "normal"
            )
            value_text = f"{v:.1f}"
            sign = "+" if signed_z >= 0 else ""
            delta_text = f"{sign}{signed_z:.1f}σ vs 평균"
        cards.append(_pipeline_card(stage, value_text, delta_text, risk))
        if i < len(PIPELINE_STAGES) - 1:
            cards.append(html.Div("→", className="pipeline-arrow"))
    return cards


def _pipeline_card(stage: dict, value_text: str, delta_text: str, risk: str) -> html.Div:
    color = RISK_COLOR[risk]
    return html.Div(
        className=f"pipeline-card pipeline-card--{risk}",
        style={"borderColor": color},
        children=[
            html.Div(
                className="pipeline-card__head",
                children=[
                    html.Span(stage["icon"], className="pipeline-card__icon"),
                    html.Span(stage["title"], className="pipeline-card__title"),
                ],
            ),
            html.Span(stage["desc"], className="pipeline-card__desc"),
            html.Div(
                className="pipeline-card__metric",
                children=[
                    html.Span(value_text, className="pipeline-card__value",
                              style={"color": color}),
                    html.Span(stage["metric_label"], className="pipeline-card__metric-label"),
                ],
            ),
            html.Span(delta_text, className="pipeline-card__delta",
                      style={"color": color}),
        ],
    )


def _gauge_figure(value: float, risk: str) -> go.Figure:
    color = RISK_COLOR[risk]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " 분", "font": {"size": 32, "color": color}},
            gauge={
                "axis": {
                    "range": [0, GAUGE_MAX],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickfont": {"color": "#94a3b8", "size": 11},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, RISK_THRESHOLDS["warning"]], "color": "rgba(16,185,129,0.18)"},
                    {"range": [RISK_THRESHOLDS["warning"], RISK_THRESHOLDS["critical"]], "color": "rgba(245,158,11,0.22)"},
                    {"range": [RISK_THRESHOLDS["critical"], GAUGE_MAX], "color": "rgba(239,68,68,0.28)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.85,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def _ts_to_clock(ts: int) -> str:
    """timestep 0-24 → 'HH:MM' (15분 간격, 00:00 시작)."""
    minutes = ts * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _detect_events(preds: list) -> dict:
    """6시간 시뮬레이션 내 risk 전환 이벤트 + 요약."""
    events = []
    prev = "normal"
    max_value = -1.0
    max_ts = 0
    first_critical = None
    critical_count = 0
    warning_count = 0

    for i, p in enumerate(preds):
        if p.value > max_value:
            max_value = p.value
            max_ts = i
        if p.risk == "critical":
            critical_count += 1
            if first_critical is None:
                first_critical = i
        if p.risk == "warning":
            warning_count += 1

        if p.risk == "critical" and prev != "critical":
            events.append({"ts": i, "kind": "enter_critical", "value": p.value})
        elif prev == "critical" and p.risk != "critical":
            events.append({"ts": i, "kind": "exit_critical", "value": p.value, "to": p.risk})
        elif p.risk == "warning" and prev == "normal":
            events.append({"ts": i, "kind": "enter_warning", "value": p.value})
        prev = p.risk

    return {
        "events": events,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "first_critical": first_critical,
        "max_value": max_value,
        "max_ts": max_ts,
        "any_critical": critical_count > 0,
        "total": len(preds),
    }


def _alerts_strip(info: dict, current_ts: int, current_pred) -> html.Div:
    summary_items = [
        _alert_kpi(
            "임계 시점",
            f"{info['critical_count']} / {info['total']}",
            "critical" if info["any_critical"] else "normal",
        ),
        _alert_kpi(
            "첫 진입",
            _ts_to_clock(info["first_critical"]) if info["first_critical"] is not None else "—",
            "critical" if info["first_critical"] is not None else "normal",
        ),
        _alert_kpi(
            "최대 지연",
            f"{info['max_value']:.1f}분 ({_ts_to_clock(info['max_ts'])})",
            _classify_risk(info["max_value"]),
        ),
        _alert_kpi(
            "현재 시점",
            f"{current_pred.value:.1f}분 ({_ts_to_clock(current_ts)})",
            current_pred.risk,
        ),
    ]

    timeline_items = []
    for ev in info["events"][:4]:
        if ev["kind"] == "enter_critical":
            timeline_items.append(_event_pill(ev["ts"], "임계 진입", ev["value"], "critical", "▲"))
        elif ev["kind"] == "exit_critical":
            timeline_items.append(_event_pill(ev["ts"], "정상 회복", ev["value"], "normal", "▼"))
        elif ev["kind"] == "enter_warning":
            timeline_items.append(_event_pill(ev["ts"], "경고 진입", ev["value"], "warning", "▲"))
    if not timeline_items:
        timeline_items = [html.Span(
            "이벤트 없음 — 6시간 동안 정상 범위", className="alerts-empty"
        )]

    return html.Div(
        className="alerts-strip__inner",
        children=[
            html.Div(className="alerts-summary", children=summary_items),
            html.Div(className="alerts-timeline", children=timeline_items),
        ],
    )


def _alert_kpi(label: str, value: str, risk: str) -> html.Div:
    return html.Div(
        className="alert-kpi",
        children=[
            html.Span(label, className="alert-kpi__label"),
            html.Span(value, className="alert-kpi__value", style={"color": RISK_COLOR[risk]}),
        ],
    )


def _event_pill(ts: int, label: str, value: float, risk: str, arrow: str) -> html.Div:
    color = RISK_COLOR[risk]
    return html.Div(
        className="event-pill",
        style={"borderColor": color},
        children=[
            html.Span(arrow, className="event-pill__arrow", style={"color": color}),
            html.Span(_ts_to_clock(ts), className="event-pill__ts"),
            html.Span(label, className="event-pill__label", style={"color": color}),
            html.Span(f"{value:.1f}분", className="event-pill__value"),
        ],
    )


def _trend_figure(values: list[float], risks: list[str], ts: int, events_info: dict | None = None) -> go.Figure:
    x = list(range(len(values)))
    y_max = max(max(values) * 1.15, RISK_THRESHOLDS["critical"] + 5)

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=RISK_THRESHOLDS["warning"],
                  fillcolor=RISK_COLOR["normal"], opacity=0.08, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_THRESHOLDS["warning"], y1=RISK_THRESHOLDS["critical"],
                  fillcolor=RISK_COLOR["warning"], opacity=0.10, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_THRESHOLDS["critical"], y1=y_max,
                  fillcolor=RISK_COLOR["critical"], opacity=0.12, line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines",
        line={"color": "#818cf8", "width": 2.5},
        fill="tozeroy", fillcolor="rgba(99,102,241,0.18)",
        hoverinfo="skip", showlegend=False,
    ))
    clocks = [_ts_to_clock(i) for i in x]
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="markers",
        marker={"size": 9, "color": [RISK_COLOR[r] for r in risks], "line": {"width": 0}},
        customdata=clocks,
        hovertemplate="%{customdata}<br>예측=%{y:.2f}분<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[ts], y=[values[ts]], mode="markers",
        marker={"size": 18, "color": RISK_COLOR[risks[ts]],
                "line": {"color": "#f8fafc", "width": 2.5}},
        hovertemplate=f"현재 시점 ({_ts_to_clock(ts)})<br>예측=%{{y:.2f}}분<extra></extra>",
        showlegend=False,
    ))

    # 이벤트 어노테이션 (임계 진입/회복) — 최대 4개
    if events_info and "events" in events_info:
        for ev in events_info["events"][:4]:
            kind = ev["kind"]
            if kind == "enter_critical":
                color = RISK_COLOR["critical"]; text = "▲ 임계 진입"; ay = -36
            elif kind == "exit_critical":
                color = RISK_COLOR["normal"]; text = "▼ 정상 회복"; ay = -32
            elif kind == "enter_warning":
                color = RISK_COLOR["warning"]; text = "▲ 경고"; ay = -28
            else:
                continue
            fig.add_annotation(
                x=ev["ts"], y=ev["value"],
                text=f"{text} {_ts_to_clock(ev['ts'])}",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, ax=0, ay=ay,
                font={"color": color, "size": 10, "family": "-apple-system, sans-serif"},
                bgcolor="rgba(15,23,42,0.92)",
                bordercolor=color, borderwidth=1, borderpad=4,
                opacity=0.95,
            )

    fig.update_layout(
        margin={"l": 50, "r": 30, "t": 10, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis={"title": "시간 (15분 간격, 0~6시간)", "gridcolor": "#1e293b", "zeroline": False,
               "tickmode": "array",
               "tickvals": [0, 4, 8, 12, 16, 20, 24],
               "ticktext": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00"]},
        yaxis={"title": "분", "gridcolor": "#1e293b", "zeroline": False, "range": [0, y_max]},
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#e2e8f0"}},
        transition={"duration": 280, "easing": "cubic-in-out"},
    )
    return fig


def _contrib_figure(contrib: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    if not contrib:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 30, "r": 20, "t": 10, "b": 30},
            annotations=[{"text": "기여도 정보 없음", "xref": "paper", "yref": "paper",
                          "x": 0.5, "y": 0.5, "showarrow": False,
                          "font": {"color": "#64748b"}}],
        )
        return fig

    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]))
    labels = [SIGNAL_LABEL.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [RISK_COLOR["critical"] if v > 0 else RISK_COLOR["normal"] for v in vals]

    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>기여 %{x:+.2f}분<extra></extra>",
    ))
    extreme = max(abs(v) for v in vals) * 1.4 or 1.0
    fig.add_vline(x=0, line_color="#475569", line_width=1)
    fig.update_layout(
        margin={"l": 130, "r": 30, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis={"title": "기여 (분, +지연증가 / -지연감소)",
               "gridcolor": "#1e293b", "zeroline": False,
               "range": [-extreme, extreme]},
        yaxis={"gridcolor": "#1e293b", "zeroline": False},
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#e2e8f0"}},
        transition={"duration": 220, "easing": "cubic-in-out"},
    )
    return fig


def _signal_cards(row: pd.Series, means: dict[str, float], seq: pd.DataFrame) -> list:
    from dash import dcc
    cards = []
    ts_idx = int(row.get("ts_rank", 0)) if "ts_rank" in row else seq.index.get_loc(row.name)
    for col in SIGNAL_COLS:
        if col not in row or pd.isna(row[col]):
            cards.append(_signal_card(SIGNAL_LABEL.get(col, col), "—", "결측", "#64748b", None))
            continue
        val = float(row[col])
        mean = means.get(col)
        if mean and mean != 0:
            pct = (val - mean) / mean * 100
            delta_color = RISK_COLOR["critical"] if pct > 15 else (
                RISK_COLOR["warning"] if pct > 0 else RISK_COLOR["normal"]
            )
            delta = f"{pct:+.0f}% vs 평균"
        else:
            delta, delta_color = "—", "#94a3b8"
        spark_values = seq[col].astype(float).tolist() if col in seq.columns else None
        spark_fig = _sparkline_figure(spark_values, ts_idx, delta_color) if spark_values else None
        cards.append(_signal_card(SIGNAL_LABEL.get(col, col), f"{val:.2f}", delta, delta_color, spark_fig))
    return cards


def _signal_card(title: str, value: str, delta: str, delta_color: str, spark_fig) -> html.Div:
    from dash import dcc
    children = [
        html.Span(title, className="signal-card__title"),
        html.Span(value, className="signal-card__value"),
        html.Span(delta, className="signal-card__delta", style={"color": delta_color}),
    ]
    if spark_fig is not None:
        children.append(html.Div(
            className="signal-card__spark",
            children=dcc.Graph(
                figure=spark_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={"height": "32px"},
            ),
        ))
    return html.Div(className="signal-card", children=children)


def _sparkline_figure(values: list[float], current_idx: int, color: str) -> go.Figure:
    x = list(range(len(values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines",
        line={"color": color, "width": 2},
        fill="tozeroy", fillcolor=color.replace(")", ",0.18)").replace("rgb", "rgba")
            if color.startswith("rgb") else _hex_to_rgba(color, 0.18),
        hoverinfo="skip", showlegend=False,
    ))
    if 0 <= current_idx < len(values):
        fig.add_trace(go.Scatter(
            x=[current_idx], y=[values[current_idx]],
            mode="markers",
            marker={"color": color, "size": 7,
                    "line": {"color": "#f8fafc", "width": 1.5}},
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 4, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
        showlegend=False,
        height=32,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(99,102,241,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _normalize_radar(col: str, val: float, radar_norm: dict) -> float:
    n = radar_norm[col]
    span = max(n["p99"] - n["p1"], 1e-6)
    return max(0.0, min(100.0, (val - n["p1"]) / span * 100))


def _radar_figure(row: pd.Series, radar_norm: dict) -> go.Figure:
    cols = [c for c in RADAR_COLS if c in radar_norm]
    labels = [RADAR_LABEL[c] for c in cols]

    cur_vals: list[float] = []
    avg_vals: list[float] = []
    for c in cols:
        v = row.get(c)
        if v is None or pd.isna(v):
            cur_vals.append(0.0)
        else:
            cur_vals.append(_normalize_radar(c, float(v), radar_norm))
        avg_vals.append(_normalize_radar(c, radar_norm[c]["mean"], radar_norm))

    cur_closed = cur_vals + [cur_vals[0]]
    avg_closed = avg_vals + [avg_vals[0]]
    labels_closed = labels + [labels[0]]

    # vertex별 색: 현재값이 평균보다 위험 방향이면 critical/warning 색
    vertex_colors = []
    vertex_lines = []
    for col, cur, avg in zip(cols, cur_vals, avg_vals):
        direction = RADAR_RISK_DIR.get(col, 1)
        diff = (cur - avg) * direction  # +면 위험 방향
        if diff >= 25:
            vertex_colors.append(RISK_COLOR["critical"])
            vertex_lines.append("#fff")
        elif diff >= 10:
            vertex_colors.append(RISK_COLOR["warning"])
            vertex_lines.append("#fff")
        else:
            vertex_colors.append("#818cf8")
            vertex_lines.append("rgba(248,250,252,0.85)")
    vertex_colors_closed = vertex_colors + [vertex_colors[0]]
    vertex_lines_closed = vertex_lines + [vertex_lines[0]]

    fig = go.Figure()

    # 위험 zone shading (옅게) — 50~75 warning, 75~100 critical
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(labels_closed), theta=labels_closed,
        fill="toself", fillcolor="rgba(239,68,68,0.10)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[75] * len(labels_closed), theta=labels_closed,
        fill="toself", fillcolor="rgba(245,158,11,0.10)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(labels_closed), theta=labels_closed,
        fill="toself", fillcolor="rgba(16,185,129,0.08)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False, hoverinfo="skip",
    ))

    # train 평균 — 점선 outline + 옅은 fill
    fig.add_trace(go.Scatterpolar(
        r=avg_closed, theta=labels_closed,
        fill="toself",
        fillcolor="rgba(148,163,184,0.06)",
        line={"color": "#94a3b8", "width": 2, "dash": "dot"},
        name="train 평균",
        hovertemplate="<b>%{theta}</b><br>평균 %{r:.0f}/100<extra></extra>",
    ))

    # 현재 — 강한 fill + 굵은 선 + vertex별 위험색 마커
    fig.add_trace(go.Scatterpolar(
        r=cur_closed, theta=labels_closed,
        fill="toself",
        fillcolor="rgba(129,140,248,0.42)",
        line={"color": "#a5b4fc", "width": 3.5, "shape": "linear"},
        mode="lines+markers",
        marker={
            "size": 14,
            "color": vertex_colors_closed,
            "line": {"color": vertex_lines_closed, "width": 2.5},
            "symbol": "circle",
        },
        name="현재",
        hovertemplate="<b>%{theta}</b><br>현재 %{r:.0f}/100<extra></extra>",
    ))

    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "tickfont": {"color": "#94a3b8", "size": 11},
                "gridcolor": "rgba(71,85,105,0.45)",
                "linecolor": "rgba(71,85,105,0.45)",
                "tickvals": [25, 50, 75, 100],
                "tickangle": 45,
            },
            "angularaxis": {
                "tickfont": {"color": "#f1f5f9", "size": 14, "family": "-apple-system, sans-serif"},
                "gridcolor": "rgba(71,85,105,0.45)",
                "linecolor": "rgba(71,85,105,0.45)",
                "rotation": 90,
                "direction": "clockwise",
            },
        },
        margin={"l": 70, "r": 70, "t": 40, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": -0.12,
            "x": 0.5, "xanchor": "center",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#f1f5f9", "size": 12},
            "itemsizing": "constant",
        },
    )
    return fig


def _top_risk_signals(row: pd.Series, radar_norm: dict) -> list:
    """위험 방향 z-score 기준 top 3."""
    deviations = []
    for c in RADAR_COLS:
        if c not in radar_norm:
            continue
        v = row.get(c)
        if v is None or pd.isna(v):
            continue
        n = radar_norm[c]
        z = (float(v) - n["mean"]) / n["std"]
        signed = z * RADAR_RISK_DIR.get(c, 1)
        deviations.append((c, float(v), signed))

    deviations.sort(key=lambda x: x[2], reverse=True)
    top = deviations[:3]

    items = []
    for rank, (col, val, signed_z) in enumerate(top, start=1):
        risk = (
            "critical" if signed_z >= 1.5 else
            "warning" if signed_z >= 0.5 else
            "normal"
        )
        action = _signal_action(col, signed_z)
        items.append(_risk_signal_item(rank, col, val, signed_z, risk, action))
    return items


def _signal_action(col: str, signed_z: float) -> str:
    if signed_z < 0.5:
        return "정상 범위 — 모니터링만"
    rules = {
        "order_inflow_15m": "주문 라우팅 일시 분산 + 핫존 작업자 1명 증원",
        "congestion_score": "AGV 우회 경로 활성화 + 일방통행 임시 해제",
        "robot_active": "유휴 로봇 투입 + 충전 큐 우선순위 재조정",
        "pack_utilization": "패킹 스테이션 추가 개방 + 출고 대기 라인 정리",
        "aisle_traffic_score": "통로 정체 — 작업자 동선 우회 안내",
        "path_optimization_score": "경로 재최적화 호출 — WMS 설정 점검",
    }
    return rules.get(col, "운영 매니저 호출 — 현장 점검")


def _risk_signal_item(rank: int, col: str, val: float, signed_z: float,
                      risk: str, action: str) -> html.Div:
    color = RISK_COLOR[risk]
    z_text = f"+{signed_z:.1f}σ" if signed_z >= 0 else f"{signed_z:.1f}σ"
    return html.Div(
        className="bottleneck-item",
        children=[
            html.Div(
                className="bottleneck-item__head",
                children=[
                    html.Span(f"#{rank}", className="bottleneck-item__rank",
                              style={"backgroundColor": color}),
                    html.Span(RADAR_LABEL.get(col, col), className="bottleneck-item__zone"),
                    html.Span(
                        z_text,
                        className="bottleneck-item__score",
                        style={"color": color},
                        title=f"현재값 {val:.2f}, 평균 대비 z-score",
                    ),
                ],
            ),
            html.Span(action, className="bottleneck-item__action"),
        ],
    )


DOW_LABELS = ["월", "화", "수", "목", "금", "토", "일"]


def _pattern_figure(pivot, cur_dow: int | None, cur_hour: int | None) -> go.Figure:
    if pivot is None:
        fig = go.Figure()
        fig.add_annotation(
            text="시간 컬럼 없음",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"color": "#64748b"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    z = pivot.values  # rows: dow 0..6, cols: hour 0..23
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=[f"{h}시" for h in range(24)],
        y=DOW_LABELS,
        colorscale=[
            [0.0, RISK_COLOR["normal"]],
            [0.5, RISK_COLOR["warning"]],
            [1.0, RISK_COLOR["critical"]],
        ],
        zmin=float(np.nanmin(z)),
        zmax=float(np.nanmax(z)),
        colorbar={
            "title": {"text": "평균<br>지연", "font": {"color": "#94a3b8", "size": 10}},
            "tickfont": {"color": "#94a3b8", "size": 10},
            "outlinewidth": 0,
            "thickness": 12,
            "len": 0.85,
            "ticksuffix": "분",
        },
        hovertemplate="%{y}요일 · %{x}<br>평균 %{z:.1f}분<extra></extra>",
        xgap=1, ygap=1,
    ))

    # 현재 시점 마커
    if cur_dow is not None and 0 <= cur_dow < 7 and cur_hour is not None and 0 <= cur_hour < 24:
        fig.add_trace(go.Scatter(
            x=[cur_hour], y=[cur_dow],
            mode="markers",
            marker={
                "size": 22,
                "color": "rgba(0,0,0,0)",
                "line": {"color": "#f8fafc", "width": 3},
                "symbol": "square",
            },
            hovertemplate=f"현재 시점의 요일·시간대<br>{DOW_LABELS[cur_dow]}요일 · {cur_hour}시<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        margin={"l": 50, "r": 30, "t": 10, "b": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis={
            "gridcolor": "#1e293b", "zeroline": False,
            "tickfont": {"size": 10, "color": "#94a3b8"},
        },
        yaxis={
            "gridcolor": "#1e293b", "zeroline": False,
            "autorange": "reversed",
            "tickfont": {"size": 11, "color": "#cbd5e1"},
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#e2e8f0"}},
    )
    return fig


def _pattern_side(pivot, overall_mean: float, cur_dow: int | None,
                  cur_hour: int | None, cur_pred: float) -> list:
    cards = []

    # 전체 평균
    cards.append(_pattern_card(
        "전체 평균", f"{overall_mean:.1f}분", "(train 데이터 기준)", "#94a3b8",
    ))

    # 현재 시점 정보
    if cur_dow is not None and cur_hour is not None and pivot is not None:
        try:
            cell = pivot.iloc[cur_dow][cur_hour] if cur_hour in pivot.columns else float("nan")
        except (IndexError, KeyError):
            cell = float("nan")
        if pd.notna(cell):
            cell_val = float(cell)
            diff = cur_pred - cell_val
            color = (
                RISK_COLOR["critical"] if diff > 5 else
                RISK_COLOR["warning"] if diff > 0 else
                RISK_COLOR["normal"]
            )
            cards.append(_pattern_card(
                f"이 시간대 평균 ({DOW_LABELS[cur_dow]}요일 {cur_hour}시)",
                f"{cell_val:.1f}분",
                f"이 시간대 train data 평균",
                "#cbd5e1",
            ))
            cards.append(_pattern_card(
                "현재 vs 시간대 평균",
                f"{diff:+.1f}분",
                "+ 면 평균 대비 더 위험" if diff > 0 else "− 면 평균 대비 양호",
                color,
            ))

    # 가장 위험한 시간대
    if pivot is not None:
        try:
            max_idx = pivot.stack().idxmax()  # (dow, hour)
            max_val = pivot.stack().max()
            cards.append(_pattern_card(
                "최고 위험 시간대",
                f"{DOW_LABELS[int(max_idx[0])]}요일 {int(max_idx[1])}시",
                f"평균 {max_val:.1f}분 — 운영 주의",
                RISK_COLOR["critical"],
            ))
        except Exception:
            pass

    return cards


def _pattern_card(title: str, value: str, hint: str, color: str) -> html.Div:
    return html.Div(
        className="pattern-card",
        children=[
            html.Span(title, className="pattern-card__title"),
            html.Span(value, className="pattern-card__value", style={"color": color}),
            html.Span(hint, className="pattern-card__hint"),
        ],
    )


def _layout_meta(layout_info_idx: pd.DataFrame, layout_id: str, cols: list[tuple[str, str]]):
    if layout_id not in layout_info_idx.index:
        return [html.Div("Layout meta 없음", className="layout-meta__cell")]
    row = layout_info_idx.loc[layout_id]
    if isinstance(row, pd.DataFrame):  # 중복 layout_id 방지
        row = row.iloc[0]
    cells = [html.Div(className="layout-meta__cell", children=[
        html.Span("LAYOUT (창고 ID)", className="layout-meta__label"),
        html.Span(str(layout_id), className="layout-meta__value"),
    ])]
    for col, label in cols:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            display = "—"
        elif col in ("layout_type", "layout_class"):
            kr = LAYOUT_TYPE_KR.get(str(v))
            display = f"{v} ({kr})" if kr else str(v)
        elif isinstance(v, float):
            display = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
        else:
            display = str(v)
        cells.append(html.Div(className="layout-meta__cell", children=[
            html.Span(label, className="layout-meta__label"),
            html.Span(display, className="layout-meta__value"),
        ]))
    return cells


def _timeline_dots(risks: list[str], ts: int):
    """25개 시점 dot — 시점별 위험도 표시 (15분 간격)."""
    dots = []
    for i, r in enumerate(risks):
        cls = "timeline-dot"
        if i == ts:
            cls += " timeline-dot--current"
        dots.append(html.Span(
            className=cls,
            style={"backgroundColor": RISK_COLOR[r]},
            title=f"{_ts_to_clock(i)} · {RISK_LABEL[r]}",
        ))
    return dots


# -- What-If 빌더 -----------------------------------------------------------

def _whatif_comparison(base, new) -> html.Div:
    delta = new.value - base.value
    delta_color = RISK_COLOR["critical"] if delta > 0 else (
        RISK_COLOR["normal"] if delta < 0 else "#94a3b8"
    )
    risk_changed = new.risk != base.risk
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")

    def block(title: str, pred, side: str) -> html.Div:
        color = RISK_COLOR[pred.risk]
        return html.Div(
            className=f"whatif-block whatif-block--{side}",
            children=[
                html.Span(title, className="whatif-block__title"),
                html.Div(
                    className="whatif-block__value",
                    children=[
                        html.Span(f"{pred.value:.1f}", style={"color": color}),
                        html.Span("분", className="whatif-block__unit"),
                    ],
                ),
                html.Span(
                    RISK_LABEL[pred.risk],
                    className="whatif-block__risk",
                    style={"backgroundColor": color},
                ),
            ],
        )

    return html.Div(
        children=[
            html.Div(
                className="whatif-comparison__row",
                children=[
                    block("현재", base, "left"),
                    html.Div(
                        className="whatif-comparison__arrow",
                        children=[
                            html.Span(arrow, style={"color": delta_color}),
                            html.Span(
                                f"{delta:+.1f}분",
                                className="whatif-comparison__delta",
                                style={"color": delta_color},
                            ),
                        ],
                    ),
                    block("What-If (가상)", new, "right"),
                ],
            ),
            html.Div(
                className="whatif-comparison__note",
                children=(
                    f"{RISK_LABEL[base.risk]} → {RISK_LABEL[new.risk]} 전환"
                    if risk_changed else "위험도 변화 없음"
                ),
                style={"color": RISK_COLOR[new.risk] if risk_changed else "#94a3b8"},
            ),
        ]
    )


def _whatif_delta_figure(base_contrib: dict, new_contrib: dict) -> go.Figure:
    keys = list(SIGNAL_COLS)
    labels = [SIGNAL_LABEL.get(k, k) for k in keys]
    base_vals = [base_contrib.get(k, 0.0) for k in keys]
    new_vals = [new_contrib.get(k, 0.0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=base_vals, orientation="h",
        name="현재",
        marker={"color": "#475569"},
        hovertemplate="%{y}<br>현재 %{x:+.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=new_vals, orientation="h",
        name="What-If (가상)",
        marker={"color": "#818cf8"},
        hovertemplate="%{y}<br>What-If %{x:+.2f}<extra></extra>",
    ))
    extreme = max([abs(v) for v in base_vals + new_vals] or [1.0]) * 1.3 or 1.0
    fig.add_vline(x=0, line_color="#475569", line_width=1)
    fig.update_layout(
        barmode="group",
        margin={"l": 130, "r": 20, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis={"title": "기여 (분)", "gridcolor": "#1e293b",
               "zeroline": False, "range": [-extreme, extreme]},
        yaxis={"gridcolor": "#1e293b", "zeroline": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0,
                "bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cbd5e1"}},
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#e2e8f0"}},
    )
    return fig


if __name__ == "__main__":
    main()
