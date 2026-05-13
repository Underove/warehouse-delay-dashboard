# 🚨 Warehouse Delay Dashboard

Dacon "스마트 물류창고 출고 지연 예측" 대회의 4-model 앙상블(LB 9.99)을 운영 의사결정 도구로 전환한 Dash 대시보드.  
운영자가 **예측 → 진단 → 시뮬레이션 → 액션**을 한 화면에서 처리할 수 있도록 설계했습니다.

---

## 화면 구성

| 섹션 | 내용 |
|------|------|
| **물류 흐름 파이프라인** | 주문 유입부터 출고까지 6단계 카드. 각 단계 신호값과 train 평균 대비 편차 |
| **현재 예측** | 선택 시점의 30분 후 평균 출고 지연 — 큰 숫자 + 위험 pill + 게이지(0~40분) |
| **시점별 추이 + 알림** | 6시간 추세선 + risk band + 임계 진입/회복 어노테이션 + KPI 4개 |
| **원인 분해** | 신호별 SHAP 기여도(분) 한국어 번역 + 핵심 신호 4개 sparkline |
| **운영 패턴 진단** | Radar — 6축 현재 vs train 평균 + 위험 신호 Top 3 권장 액션 |
| **What-If 시뮬레이션** | 4개 신호 슬라이더 → 즉시 재예측, 현재 vs 가상 비교 + 기여도 변화 차트 |

위험도 기준: **정상 < 22분 / 경고 22~30분 / 임계 ≥ 30분**

---

## 모델 구성

4-model 앙상블, 5-fold 평균, log1p 타깃:

| 모델 | 가중치 |
|------|--------|
| LightGBM | 0.2806 |
| CatBoost | 0.1702 |
| BiGRU | 0.2940 |
| BiGRU + Multi-Head Attention | 0.2552 |

- **추론**: Tree 모델은 168-feature 캐시(`test_features.parquet`), 시퀀스 모델은 `test_seq.npy`(shape `2000×25×173`) — ID 기반 O(1) lookup
- **SHAP**: `test_shap.npy`(50000×168) 사전 계산, 피처명 한국어 번역 표시
- **What-If**: 슬라이더 변경 시 tree 모델만 row 단위 재추론, 시퀀스는 baseline 캐시 유지
- **Fallback**: 모델 파일이 없으면 `MockPredictor`(z-score 더미) 자동 전환

---

## 기술 스택

- Python 3.12 / Dash 4.1 / Plotly 5.9
- pandas, numpy, scikit-learn, SHAP 0.45+
- LightGBM 4.6 / CatBoost 1.2 / PyTorch 2.3
- dash-bootstrap-components

---

## 폴더 구조

```
warehouse-dashboard/
├── config.py                # 경로, 위험 임계값, SEQ_LEN
├── requirements.txt
├── src/
│   ├── app.py               # Dash 엔트리, 콜백, 차트/카드 빌더
│   ├── inference.py         # EnsemblePredictor (LGB+CB+GRU+Attn), MockPredictor
│   ├── preprocess.py        # FE 파이프라인 (lag/rolling/sc_mean/interaction/kNN)
│   ├── data_loader.py       # CSV 로드 + 무결성 리포트
│   └── components/
│       └── layout.py        # 페이지 레이아웃
├── scripts/
│   ├── prepare_features.py  # test_features.parquet 생성
│   ├── prepare_sequences.py # test_seq.npy + seq_scaler.pkl 생성
│   └── prepare_shap.py      # test_shap.npy 생성
├── assets/
│   └── style.css            # 다크 테마
├── models/                  # (gitignore) weights.json, fold 파일들
└── data/                    # (gitignore) 캐시 파일들
```

CSV 데이터는 부모 폴더(`warehouse-dashboard/` 상위)에 위치, `config.py`에서 상대 경로로 참조.

---

## 셋업 & 실행

CSV 파일(`train.csv`, `test.csv`, `layout_info.csv`, `sample_submission.csv`)은 프로젝트 루트의 부모 폴더에 있어야 합니다.

```bash
cd warehouse-dashboard

# 의존성 설치
pip install -r requirements.txt

# 캐시 생성 (최초 1회)
python scripts/prepare_features.py   # data/test_features.parquet (~30초)
python scripts/prepare_sequences.py  # data/test_seq.npy + models/seq_scaler.pkl (~1분)
python scripts/prepare_shap.py       # data/test_shap.npy (~2~3분)

# 실행
python src/app.py
# → http://127.0.0.1:8050
```

모델 파일(`models/*.txt`, `*.cbm`, `*.pt`, `weights.json`)은 gitignore 대상.  
파일이 없으면 MockPredictor로 동작해 UI는 정상 표시됩니다.

정상 부팅 로그:
```
[inference] tree models loaded  (lgb×5, cb×5)
[inference] seq models loaded   shape=(2000, 25, 173)
[inference] shap cache loaded   shape=(50000, 168)
Dash is running on http://127.0.0.1:8050/
```

---

## 라이선스

개인 포트폴리오 / 대회 부수 자료. 대회 데이터는 Dacon 정책을 따르며 `data/`, `models/`, `*.csv` 일체 저장소에 포함되지 않습니다.
