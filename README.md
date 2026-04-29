# Warehouse Delay Dashboard

스마트 창고 출고 지연 예측 모델을 **운영 의사결정 도구**로 변환하는 Dash 대시보드. Dacon "스마트 물류창고 출고 지연 예측" 대회의 5모델 앙상블(LB 9.99)을 백엔드로, 운영자가 30분 후 예측을 보고 바로 액션을 취할 수 있는 6-섹션 화면을 제공합니다.

> 핵심 메시지: **예측 → 진단 → 시뮬레이션 → 액션**을 한 페이지에.

---

## 한눈에 보는 화면 구성

| # | 섹션 | 무엇을 보여주나 |
|---|---|---|
| 1 | **현재 예측** | 선택한 시점의 30분 후 평균 출고 지연 — 큰 숫자, 위험 pill, 게이지(0~40분), 임계까지 여유 |
| 2 | **시점별 추이 + 알림** | 25 timestep 추세선 + risk band 배경 + 임계 진입/회복 자동 어노테이션. 위 strip에 KPI 4장 + 이벤트 pill |
| 3 | **원인 분해 + 현재 신호** | 신호별 기여도 (mock: z·1.5, 향후 SHAP 교체) + 핵심 신호 4장 sparkline |
| 4 | **운영 패턴 진단** | Radar — 6개 핵심 신호의 현재 vs train 평균 비교 + 위험 신호 Top 3 권장 액션 |
| 5 | **시간대별 패턴** | 요일 × 시간대(7×24) 평균 지연 heatmap, 현재 시점 흰 테두리 셀 |
| 6 | **What-If 시뮬레이션** | 4개 신호 슬라이더 → 즉시 재예측, 현재 vs 가상 비교 카드 + 기여도 변화 |

페이지 상단에는 워크플로우 5단계 가이드와 위험도 범례(정상 < 15분, 경고 15~22분, 임계 ≥ 22분)가 항상 표시됩니다.

---

## 데이터 / 모델

- **데이터**: 대회 train(250k×94) / test(50k×93) / layout_info(300×15)
  - 좌표 정보 없음 → 격자 평면도 대신 **신호 기반 radar/시간대 heatmap**으로 도메인 시각화
  - 시퀀스: 한 scenario당 25 timestep 일관
- **추론**:
  - `MockPredictor` (현재 사용) — train 평균 대비 z-score를 합산. UI 개발용
  - `EnsemblePredictor` — `models/weights.json` + 5-fold(LGB/XGB/CB/GRU+Attn) 도착 시 `load_predictor()`가 자동 전환
- **위험 임계**: normal < 15분 / warning 15~22분 / critical ≥ 30분 (정확히는 `config.py` 참고)

---

## 기술 스택

- Python 3.12 / Dash 4.1 / Plotly 6.7
- pandas, numpy, scikit-learn
- LightGBM 4.6 / XGBoost 3.2 / CatBoost 1.2 / PyTorch 2.11 / SHAP 0.51 (모델 통합용)
- dash-bootstrap-components (Select 다크 테마용)

UI 노트:
- `dcc.Dropdown` (react-select v5 emotion)은 다크 테마 적용이 까다로워 **`dbc.Select`**(native HTML)로 교체
- `dcc.Slider` tooltip은 dash 4.x에서 hide 옵션이 무시되어 **`dcc.Input(type='range')`**(native HTML range)로 교체

---

## 폴더 구조

```
warehouse-dashboard/
├── config.py                # 경로 / 임계값 / SEQ_LEN
├── requirements.txt
├── src/
│   ├── app.py               # Dash 엔트리, 콜백, 차트/카드 빌더
│   ├── data_loader.py       # csv 로드 + 무결성 리포트
│   ├── inference.py         # MockPredictor + EnsemblePredictor stub
│   └── components/
│       └── layout.py        # 페이지 레이아웃 (intro strip, 6 sections)
├── assets/
│   └── style.css            # 다크 테마 + 컴포넌트별 스타일
├── models/                  # (gitignore) 학습 산출물 — weights.json, fold pkl 등
├── data/                    # (gitignore) 로컬 캐시 (옵션)
└── README.md
```

데이터 csv는 부모 폴더(`~/Desktop/open/`)에 두고 `config.py`에서 상대 경로로 참조.

---

## 셋업 & 실행

```bash
cd warehouse-dashboard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (옵션) 데이터 무결성 검증
python src/data_loader.py

# 대시보드 띄우기
python src/app.py
# → http://127.0.0.1:8050/
```

기대 출력 (`data_loader.py`):
- train (250000, 94), test (50000, 93)
- layout_info 300 (train 250 / test 100 / test-only 50)
- TARGET `avg_delay_minutes_next_30m` mean ≈ 18.96
- SEQ_LEN 25 일관

---

## 진행 현황

- [x] **Phase 0** — 폴더 구조, config, data_loader, venv, 데이터 무결성 검증
- [x] **Phase 1** — Dash 골격: 헤더, 컨트롤, hero/gauge, 추세, 기여도, 신호 sparkline, layout 메타
- [x] **Phase 2** — What-If Simulator: 4개 신호 슬라이더, 현재 vs 가상 비교, 기여도 변화
- [x] **Phase 3** — 운영 패턴 진단: Radar (현재 vs train 평균) + 위험 신호 Top 3 권장 액션
- [x] **Phase 4** — 알림 시스템: 임계 진입/회복 이벤트 감지, 추세 차트 어노테이션, KPI strip + 이벤트 pill
- [x] **Phase 5** — 시간대 패턴: 요일×시간(7×24) heatmap, 현재 시점 마커, 시간대 평균 비교 카드
- [x] UI 가이드: 워크플로우 5단계 strip, 위험도 범례, 섹션 번호+한 줄 안내, 한국어 라벨
- [ ] **Phase 6** — 배포 (Render/Railway)
- [ ] EnsemblePredictor 구현 (모델 파일 도착 시)
- [ ] SHAP 통합 (Mock 기여도 → 실제 SHAP value)
- [ ] 실시간 stream API (대회 종료 후, 면접 답변용)

---

## 모델 파일 컨벤션 (도착 시 자동 인식)

```
models/
├── weights.json            # {"lgb": 0.32, "xgb": 0.24, "cb": 0.43, "gru": ...}
├── lgb_fold0.txt … fold4.txt
├── xgb_fold0.json … fold4.json
├── cb_fold0.cbm  … fold4.cbm
└── gru_fold0.pt  … fold4.pt
```

`load_predictor(ref_df)`가 `weights.json` 존재 여부로 `MockPredictor` ↔ `EnsemblePredictor` 자동 전환합니다.

---

## 라이선스

개인 포트폴리오 / 대회 부수 자료. 대회 데이터 라이선스는 Dacon 정책을 따름 — `data/`, `*.csv`, `models/*` 일체 푸시되지 않습니다.
