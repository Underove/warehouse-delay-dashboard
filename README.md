# Warehouse Delay Dashboard

스마트 창고 출고 지연 예측 모델(Dacon 대회 5모델 앙상블)을 운영 의사결정 도구로 변환하는 Dash 대시보드.

## 구조
```
warehouse-dashboard/
├── config.py            # 경로/임계값/시퀀스 길이
├── requirements.txt
├── src/
│   ├── data_loader.py   # Phase 0 데이터 검증
│   └── components/      # Dash 컴포넌트 (Phase 1~)
├── models/              # 학습 산출물 (gitignore)
├── data/                # (옵션) 로컬 캐시 — 기본은 부모 폴더 참조
└── assets/              # css, 정적 자산
```

## Phase 0 셋업

```bash
cd ~/Desktop/open/warehouse-dashboard
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data_loader.py
```

기대 출력: train (250000, ...), test (50000, ...), layout_info (300, ...), test-only layouts 50개.

## Phase 진행
- [x] Phase 0 — 폴더/설정/데이터 검증
- [ ] Phase 1 — Dash 골격 + 시간 슬라이더 + 5모델 추론 통합
- [ ] Phase 2 — What-If Simulator (핵심)
- [ ] Phase 3 — 창고 schematic + 혼잡도 heatmap
- [ ] Phase 4 — KPI 카드 + 알림 + bottleneck 진단
- [ ] Phase 5 — 배포 (Render/Railway)

## 데이터/모델 노트
- layout_info에 좌표 없음 → 평면도는 schematic representation으로 처리
- 학습은 Colab(GPU)에서, 추론만 로컬에서 (모델 파일 다운로드 후 `models/`)
- 위험 임계: normal 15분 / warning 22분 / critical 30분
