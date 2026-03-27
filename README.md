# 🔎 AGTrack
**실시간 다중 객체 추적 및 인구통계 분석 시스템**

개인 프로젝트 | 2025.05.13 ~ 2025.05.30

---

## 🎯 핵심 성과
- **처리 시간 76.7% 단축**: 30초 영상 기준 30분 → 7분
- **Re-ID 안정화**: Cosine similarity 기반 로직 직접 설계로 ID switching 감소
- **정확도-속도 Trade-off 최적화**: KF 파라미터 실험 기반 튜닝

---

## 📌 프로젝트 개요
ByteTrack으로 다수의 사람을 실시간 추적하고, MiVOLO로 각 객체의 나이와 성별을 추정하는 파이프라인 구축.
매장 환경에서 시간대별 고객 인구통계(나이·성별·유입 패턴)를 분석하고, Gradio 대시보드로 시각화하는 것을 목표로 함.

---

## Dataset
이번 프로젝트에서는 AI Hub에서 제공하는 멀티 센서 동선 추적 데이터(Multi-sensor Movement Tracking Data)를 활용하였습니다.

실험 및 개발을 신속하게 진행하기 위해, 해당 데이터셋 중 테스트 데이터만을 사용하였습니다.

---

## Demo
<img width="1273" alt="image" src="https://github.com/user-attachments/assets/d49a16e4-dc40-4af8-af01-4bf509cce8db" />
<img width="1277" alt="image" src="https://github.com/user-attachments/assets/622c24f6-31e6-4892-9a0e-767f51f99ab9" />

---
## 🛠 기술 스택
- **Tracking**: ByteTrack, Kalman Filter, BoostTrack++
- **Age/Gender**: MiVOLO
- **Optimization**: Cosine Similarity 기반 Re-ID, Similarity Matrix 병렬화
- **Visualization**: Gradio Dashboard
- **Data**: AI Hub 멀티센서 동선 추적 데이터

---

## 💡 기술적 의사결정

### 왜 ByteTrack인가?
- BoostTrack++ 논문 기반으로 MOT 구조를 학습하고 분석한 후, occlusion 상황에서 안정적인 ByteTrack을 현장 영상에 적합한 베이스로 선택

### Re-ID 로직 직접 설계
- 기존 방식의 ID switching 문제를 원인 분석 후 직접 해결
- Cosine similarity threshold 0.9, sliding window 10프레임으로 안정화
- 모든 embedding을 행렬화하여 내적 연산으로 최적화 → Re-ID 오버헤드 최소화

### KF 파라미터 튜닝
- TRACK_THRESH / MATCH_THRESH / ASPECT_RATIO_THRESH 실험 기반 튜닝
- 정확도와 속도 사이의 Trade-off를 고려한 최적 파라미터 조합 도출

---

## 📊 결과
| 항목 | Before | After |
|------|--------|-------|
| 처리 시간 (30초 영상) | 30분 | 7분 |
| 단축률 | - | 76.7% |
| ID switching | 높음 | Re-ID 로직으로 감소 |

---

## 🔍 주요 기능
- **실시간 다중 객체 추적**: ByteTrack 기반 다수 인원 동시 추적
- **나이·성별 추정**: MiVOLO를 활용한 실시간 인구통계 분석
- **고객 유입 분석**: 시간대별 고객 나이 분포 및 유입 패턴 시각화
- **Gradio 대시보드**: 실시간 통계 제공 및 실험 리포트 시각화

---

## ⚠️ 한계 및 향후 개선
- 완전한 실시간(< 1초)에는 미달
- 시간 제약으로 구조적 경량화 미진행
- 향후 모델 경량화 및 E2E 파이프라인 관점에서 추론 구조 최적화 예정

---

## 🗂 프로젝트 구조
```
AGTrack/
├── MiVOLO/          # MiVOLO 모델
├── ByteTrack/       # ByteTrack 모델
├── data/            # 멀티센서 동선 추적 데이터
├── AGPredictor.py   # 나이·성별 추정
├── tracking.py      # 다중 객체 추적
├── inference.py     # 추론 파이프라인
├── app.py           # Gradio 데모
└── README.md
```



