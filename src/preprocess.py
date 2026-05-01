"""
노트북 7-2의 피처 엔지니어링 파이프라인을 추론용으로 옮긴 모듈.

호출 순서:
    train_raw, test_raw, layout_raw = load_*()
    train_fe, test_fe = build_features(train_raw, test_raw, layout_raw)

train_fe는 kNN 타겟 인코딩 계산용으로만 필요 (test 추론에는 test_fe만 사용).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TARGET

ID_COLS = ["ID", "layout_id", "scenario_id"]

LAG_COLS = [
    "order_inflow_15m", "congestion_score", "robot_active",
    "battery_mean", "max_zone_density", "pack_utilization",
    "aisle_traffic_score", "path_optimization_score",
]

SC_MEAN_COLS = [
    "low_battery_ratio", "battery_mean", "robot_idle",
    "order_inflow_15m", "robot_charging", "max_zone_density", "congestion_score",
]

FEATURES_TO_DROP = [
    "task_reassign_15m", "charge_queue_length", "blocked_path_15m",
    "fault_count_15m", "avg_charge_wait", "avg_recovery_time",
    "lt_narrow", "lt_hybrid", "lt_hub_spoke",
    "delay_acceleration", "near_collision_15m",
]

LEAK_FEATURES = [
    "lt_grid", "low_pack_capacity", "detour_probability",
    "robot_charging", "robot_active",
]


def merge_layout(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    layout_oh = pd.get_dummies(layout, columns=["layout_type"], prefix="lt")
    return df.merge(layout_oh, on="layout_id", how="left")


def add_id_num_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id_num"] = df["ID"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["scenario_id", "id_num"]).reset_index(drop=True)
    df["ts_rank"] = df.groupby("scenario_id").cumcount()
    return df


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LAG_COLS:
        g = df.groupby("scenario_id")[col]
        df[f"{col}_lag1"] = g.shift(1)
        df[f"{col}_lag2"] = g.shift(2)
        df[f"{col}_delta"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]
        df[f"{col}_roll4_mean"] = g.transform(lambda x: x.rolling(4, min_periods=1).mean())
        df[f"{col}_pure_past_roll4"] = g.transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )
    return df.fillna(0)


def add_sc_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in SC_MEAN_COLS:
        df[f"{c}_sc_mean"] = df.groupby("scenario_id")[c].transform("mean")
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_speed_ratio"] = df["order_inflow_15m"] / (df["order_inflow_15m_sc_mean"] + 1e-6)
    df["packing_pressure"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1e-6)
    df["low_pack_capacity"] = (df["pack_station_count"] <= 6).astype(int)
    df["is_full_density"] = (df["max_zone_density"] >= 0.98).astype(int)

    df["delay_acceleration"] = df.groupby("scenario_id")["congestion_score"].diff().fillna(0)
    df["order_cumsum"] = df.groupby("scenario_id")["order_inflow_15m"].cumsum()
    df["order_roll4_sum"] = df.groupby("scenario_id")["order_inflow_15m"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    df["accel_roll3_mean"] = df.groupby("scenario_id")["delay_acceleration"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df["true_congestion_penalty"] = df["congestion_score"] * (
        df["intersection_count"] / (df["aisle_width_avg"] + 1e-6)
    )
    df["detour_probability"] = df["congestion_score"] * df["one_way_ratio"]
    df["packing_bottleneck"] = df["order_cumsum"] / (df["pack_station_count"] + 1e-6)
    df["robot_density"] = df["robot_active"] / (df["floor_area_sqm"] + 1e-6)
    df["charger_queue_pressure"] = df["robot_active"] / (df["charger_count"] + 1e-6)
    df["backlog_per_robot"] = df["order_cumsum"] / (df["robot_active"] + 1e-6)
    df["bottleneck_intensity"] = df["congestion_score"] / (df["aisle_width_avg"] + 1e-6)

    layout_pen = 1.0
    if "lt_narrow" in df.columns:
        layout_pen = layout_pen + 0.15 * df["lt_narrow"]
        df["congestion_x_narrow"] = df["congestion_score"] * df["lt_narrow"]
    if "lt_hub_spoke" in df.columns:
        layout_pen = layout_pen + 0.10 * df["lt_hub_spoke"]
        df["order_pressure_x_hub_spoke"] = df["order_inflow_15m"] * df["lt_hub_spoke"]
    df["layout_structural_risk"] = df["bottleneck_intensity"] * layout_pen

    df["battery_recovery_burden"] = (df["robot_charging"] * 50) / (df["robot_active"] + 1e-6)
    df["hrc_load_factor"] = df["order_cumsum"] / (df["robot_active"] + df["pack_station_count"] + 1e-6)

    if "wms_response_time_ms" in df.columns:
        df["wms_load_pressure"] = df["wms_response_time_ms"] * df["order_inflow_15m"]
    if "agv_task_success_rate" in df.columns:
        df["agv_congestion_risk"] = (1 - df["agv_task_success_rate"]) * df["congestion_score"]
    if "path_optimization_score" in df.columns:
        df["path_congestion_risk"] = (1 - df["path_optimization_score"]) * df["congestion_score"]
    if "staff_on_floor" in df.columns:
        df["staff_order_pressure"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)
    if "outbound_truck_wait_min" in df.columns:
        df["truck_backlog_pressure"] = df["outbound_truck_wait_min"] * df["order_cumsum"]

    if "intersection_wait_time_avg" in df.columns:
        df["total_intersection_delay"] = df["intersection_wait_time_avg"] * df["intersection_count"]

    return df


def build_layout_knn_features(
    train: pd.DataFrame, test: pd.DataFrame, layout: pd.DataFrame, k: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """노트북의 `build_layout_knn_features`를 그대로 옮김. layout-level 3 features."""
    layout_stats = train.groupby("layout_id")[TARGET].agg(["mean", "median", "std"]).reset_index()
    layout_stats.columns = ["layout_id", "lyt_tgt_mean", "lyt_tgt_median", "lyt_tgt_std"]

    train_lyt = layout_stats.merge(layout, on="layout_id", how="left")
    test_lyt = pd.DataFrame({"layout_id": test["layout_id"].unique()}).merge(
        layout, on="layout_id", how="left"
    )

    numeric_cols = [
        "aisle_width_avg", "intersection_count", "one_way_ratio",
        "pack_station_count", "charger_count", "layout_compactness",
        "zone_dispersion", "robot_total", "building_age_years",
        "floor_area_sqm", "ceiling_height_m",
        "fire_sprinkler_count", "emergency_exit_count",
    ]
    all_lyt = pd.concat(
        [
            train_lyt[["layout_id", "layout_type"] + numeric_cols],
            test_lyt[["layout_id", "layout_type"] + numeric_cols],
        ],
        ignore_index=True,
    )
    all_lyt = pd.get_dummies(all_lyt, columns=["layout_type"], dtype=float)

    feat_cols = [c for c in all_lyt.columns if c != "layout_id"]
    X_all = StandardScaler().fit_transform(all_lyt[feat_cols].fillna(0))

    n_tr = len(train_lyt)
    X_tr = X_all[:n_tr]
    X_te = X_all[n_tr:]
    tgt_arr = train_lyt["lyt_tgt_mean"].values

    nn_tr = NearestNeighbors(n_neighbors=k + 1).fit(X_tr)
    _, idx_tr = nn_tr.kneighbors(X_tr)
    idx_tr = idx_tr[:, 1:]
    train_lyt["knn_target_mean"] = tgt_arr[idx_tr].mean(axis=1)
    train_lyt["knn_target_std"] = tgt_arr[idx_tr].std(axis=1)
    train_lyt["knn_target_max"] = tgt_arr[idx_tr].max(axis=1)

    nn_te = NearestNeighbors(n_neighbors=k).fit(X_tr)
    _, idx_te = nn_te.kneighbors(X_te)
    test_lyt["knn_target_mean"] = tgt_arr[idx_te].mean(axis=1)
    test_lyt["knn_target_std"] = tgt_arr[idx_te].std(axis=1)
    test_lyt["knn_target_max"] = tgt_arr[idx_te].max(axis=1)

    knn_cols = ["knn_target_mean", "knn_target_std", "knn_target_max"]
    return train_lyt[["layout_id"] + knn_cols], test_lyt[["layout_id"] + knn_cols]


def build_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame, layout_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw → 학습/추론에 쓸 수 있는 fully-featured 데이터프레임."""
    train = merge_layout(train_raw, layout_raw)
    test = merge_layout(test_raw, layout_raw)

    train = add_id_num_and_sort(train)
    test = add_id_num_and_sort(test)

    train = add_lag_rolling(train)
    test = add_lag_rolling(test)

    train = add_sc_mean(train)
    test = add_sc_mean(test)

    train = add_interactions(train)
    test = add_interactions(test)

    train_knn, test_knn = build_layout_knn_features(train, test, layout_raw)
    train = train.merge(train_knn, on="layout_id", how="left")
    test = test.merge(test_knn, on="layout_id", how="left")

    return train, test


SIGNAL_DEPENDENT_COLS = (
    "packing_pressure",
    "true_congestion_penalty",
    "detour_probability",
    "robot_density",
    "charger_queue_pressure",
    "backlog_per_robot",
    "bottleneck_intensity",
    "congestion_x_narrow",
    "order_pressure_x_hub_spoke",
    "layout_structural_risk",
    "battery_recovery_burden",
    "hrc_load_factor",
    "wms_load_pressure",
    "agv_congestion_risk",
    "path_congestion_risk",
    "staff_order_pressure",
)


def recompute_signal_dependent(df: pd.DataFrame) -> pd.DataFrame:
    """
    What-If: 4 signal cols(`order_inflow_15m`, `congestion_score`, `robot_active`,
    `pack_utilization`) 가 바뀐 후 그 값에만 의존하는 per-row 파생 피처를 다시 계산.
    lag/rolling/sc_mean/cumsum 등 다중 timestep 의존 피처는 baseline 값 유지 (단일 row
    What-If의 한계).
    """
    df = df.copy()
    df["packing_pressure"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1e-6)
    df["true_congestion_penalty"] = df["congestion_score"] * (
        df["intersection_count"] / (df["aisle_width_avg"] + 1e-6)
    )
    df["detour_probability"] = df["congestion_score"] * df["one_way_ratio"]
    df["robot_density"] = df["robot_active"] / (df["floor_area_sqm"] + 1e-6)
    df["charger_queue_pressure"] = df["robot_active"] / (df["charger_count"] + 1e-6)
    df["backlog_per_robot"] = df["order_cumsum"] / (df["robot_active"] + 1e-6)
    df["bottleneck_intensity"] = df["congestion_score"] / (df["aisle_width_avg"] + 1e-6)

    layout_pen = 1.0
    if "lt_narrow" in df.columns:
        layout_pen = layout_pen + 0.15 * df["lt_narrow"]
        df["congestion_x_narrow"] = df["congestion_score"] * df["lt_narrow"]
    if "lt_hub_spoke" in df.columns:
        layout_pen = layout_pen + 0.10 * df["lt_hub_spoke"]
        df["order_pressure_x_hub_spoke"] = df["order_inflow_15m"] * df["lt_hub_spoke"]
    df["layout_structural_risk"] = df["bottleneck_intensity"] * layout_pen

    df["battery_recovery_burden"] = (df["robot_charging"] * 50) / (df["robot_active"] + 1e-6)
    df["hrc_load_factor"] = df["order_cumsum"] / (
        df["robot_active"] + df["pack_station_count"] + 1e-6
    )

    if "wms_response_time_ms" in df.columns:
        df["wms_load_pressure"] = df["wms_response_time_ms"] * df["order_inflow_15m"]
    if "agv_task_success_rate" in df.columns:
        df["agv_congestion_risk"] = (1 - df["agv_task_success_rate"]) * df["congestion_score"]
    if "path_optimization_score" in df.columns:
        df["path_congestion_risk"] = (1 - df["path_optimization_score"]) * df["congestion_score"]
    if "staff_on_floor" in df.columns:
        df["staff_order_pressure"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)

    return df


def get_optimized_cols(train_fe: pd.DataFrame) -> list[str]:
    """노트북의 optimized_feature_cols 재현 (kNN 포함)."""
    base = [c for c in train_fe.columns if c not in ID_COLS + [TARGET, "id_num"]]
    return [c for c in base if c not in FEATURES_TO_DROP]


def get_reduced_cols(train_fe: pd.DataFrame) -> list[str]:
    """Tree 모델용 — optimized 에서 leak 5개 추가 drop."""
    return [c for c in get_optimized_cols(train_fe) if c not in LEAK_FEATURES]
