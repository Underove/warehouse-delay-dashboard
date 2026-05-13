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

> **전제**: 프로젝트 루트(`warehouse-dashboard/`)의 부모 폴더에 `train.csv`, `test.csv`, `layout_info.csv`가 있어야 합니다.

```bash
cd warehouse-dashboard

# 1. 의존성 설치 (Python 3.11+ 권장, conda 환경 또는 venv)
pip install -r requirements.txt

# 2. 피처 캐시 생성 (최초 1회, ~30초)
python scripts/prepare_features.py   # data/test_features.parquet
python scripts/prepare_sequences.py  # data/test_seq.npy + models/seq_scaler.pkl

# 3. 대시보드 실행
python src/app.py
# → http://127.0.0.1:8050/
```

> **모델 파일**(`models/*.txt`, `*.cbm`, `*.pt`, `weights.json`)은 gitignore 대상입니다.  
> 파일이 없으면 MockPredictor(더미 예측)로 자동 fallback되므로 UI 확인은 가능합니다.

기대 부팅 로그:
```
[inference] seq models loaded  shape=(2000, 25, 173)
[boot] layout meta cols: ['layout_type', ...]
Dash is running on http://127.0.0.1:8050/
```

---

## 진행 현황

- [x] **Phase 0** — 폴더 구조, config, data_loader, 데이터 무결성 검증
- [x] **Phase 1** — Dash 골격: 헤더, 컨트롤, hero/gauge, 추세, 기여도, 신호 sparkline, layout 메타
- [x] **Phase 2** — What-If Simulator: 4개 신호 슬라이더, 현재 vs 가상 비교, 기여도 변화
- [x] **Phase 3** — 운영 패턴 진단: Radar (현재 vs train 평균) + 위험 신호 Top 3 권장 액션
- [x] **Phase 4** — 알림 시스템: 임계 진입/회복 이벤트 감지, 추세 차트 어노테이션, KPI strip + 이벤트 pill
- [x] **Phase 5(제거)** — day_of_week × shift_hour heatmap (시나리오 내 dow 불일치 확인 후 제거)
- [x] **Phase B** — 4-model EnsemblePredictor: LGB + CB + BiGRU + BiGRU+Attn 5-fold 앙상블
  - 가중치 합 1.0 그대로 사용 (lgb 0.2806 / cb 0.1702 / gru 0.2940 / attn 0.2552)
  - 시퀀스 모델은 scenario 단위 forward 후 캐시 → What-If는 tree 모델만 반응
- [x] UI 가이드: 워크플로우 4단계 strip, 위험도 범례, 섹션 번호 안내, 한국어 라벨
- [ ] SHAP 통합 (현재 z-score Mock 기여도 사용 중)
- [ ] 배포 (Render/Railway)

---

## 모델 파일 구조

```
models/
├── weights.json              # 앙상블 가중치 (lgb/cb/gru/attn)
├── lgb_fold0.txt … fold4.txt
├── cb_fold0.cbm  … fold4.cbm
├── gru_fold0.pt  … fold4.pt
├── attn_fold0.pt … fold4.pt
└── seq_scaler.pkl            # StandardScaler (prepare_sequences.py 생성)
```

`load_predictor(ref_df)`가 `weights.json` + `test_features.parquet` 존재 여부로 `MockPredictor` ↔ `EnsemblePredictor` 자동 전환합니다.

---

## 라이선스

개인 포트폴리오 / 대회 부수 자료. 대회 데이터 라이선스는 Dacon 정책을 따름 — `data/`, `*.csv`, `models/*` 일체 푸시되지 않습니다.
