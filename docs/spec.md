# Skycolor Locator Spec (Single Source of Truth)

> 이 문서는 **서비스 목표, 입력/출력 계약, 오케스트레이션(데이터/모델/검색) 기준**을 한 곳에서 고정하는 *단일 소스*입니다.
> 코드/테스트/인프라 변경은 본 문서의 계약(Contract)과 가이드라인을 **우선** 준수해야 합니다.

## 0. 한 줄 정의
실시간(또는 준실시간) 위성/기상 관측과 정적/저주기 갱신 지표 정보를 바탕으로, 주어진 시간과 위치에서 **지상 관측자(ground-based)** 관점의 **하늘+땅 색 분포 곡선(Color Signature)** 을 예측하고, 사용자가 원하는 색감(타겟 시그니처)에 가장 가까운 **지구상의 위치/시간 후보**를 찾아주는 서비스.

---

## 1. 사용자 가치(Use Cases)

### 1.1 “지금 내가 원하는 색감이 있는 곳”
- 입력: (now) + 원하는 색감(레퍼런스 이미지/팔레트/히스토그램)
- 출력: 지금 시각 기준 전지구 후보 위치 Top-K(위도/경도) + 예상 색 시그니처 + 설명(왜 이 색이 나왔는지)

### 1.2 “특정 시간의 특정 위치가 어떤 색감인가”
- 입력: (time_utc, lat, lon)
- 출력: 해당 조건의 예측 시그니처(하늘/지표 분리 곡선) + 품질/불확실도 + 주요 기여 요인(태양고도/구름/오존/에어로졸/지표타입)

### 1.3 “촬영/여행 계획용”
- 입력: 시간 범위(예: 2026-02-14 09:00~18:00 UTC) + 원하는 색감
- 출력: 시간-공간 후보(위치/시각) + 트렌드(시간에 따라 시그니처가 어떻게 변할지)

---

## 2. 문제 정의

### 2.1 입력 x
서비스 핵심 입력은 다음의 튜플로 정의한다.

- `x = (time_utc, lat_deg, lon_deg, camera_profile?)`
  - `time_utc`: timezone-aware UTC datetime
  - `lat_deg, lon_deg`: WGS84
  - `camera_profile`(선택): sRGB 가정이 기본. 추가로 WB/노출/렌즈/시야각/방향 등을 확장 가능.

### 2.2 출력 y
출력은 **색 분포 곡선(Color Signature)** 이며, 최소 구성은 다음과 같다.

- `Sky Hue Histogram` (N bins)
- `Ground Hue Histogram` (N bins)
- `Signature Vector = concat(sky_hist, ground_hist)` (2N dim)
- 메타/설명: 태양 위치, 대기/구름 상태 요약, 품질 플래그

> **중요:** 우리는 “렌더링된 사진 한 장”을 반드시 출력할 필요는 없다. 서비스의 핵심은 전지구 검색이 가능한 **시그니처 벡터**이며, 이미지 합성은 설명/미리보기 용도로 2차 기능이다.

---

## 3. 핵심 개념과 데이터 구분

### 3.1 EarthState
`EarthState(time, tile_or_point)`는 예측에 필요한 상태 벡터.

- **실시간/준실시간(Realtime) 관측치**: 위성/기상/재분석 등
  - 구름: cloud fraction, cloud optical depth(가능하면)
  - 에어로졸: AOD(예: 550nm)
  - 오존: TOC(Dobson Unit)
  - 수증기/습도/가시거리/기압(가능하면)

- **저주기 갱신(Periodic Constants)**: 특정 주기로 업데이트되는 상수
  - 건물/도시 재질 색, 지표 반사도(계절 평균), 토지피복 클래스 비율 등

- **정적(Static Constants)**: 물리적으로 고정된 상수
  - 산란/흡수 계수(모델 파라미터), 색공간 변환, 기본 태양 스펙트럼 근사 상수 등

### 3.2 Color Signature
- 정의: 사람이 지상에서 바라보는 장면을 **Sky**(하늘)와 **Ground**(지표)로 분리했을 때의 색 분포.
- 기본 구현: Hue histogram (가중치: S×V 또는 S)
- 확장 구현(선택):
  - (H,S,V) 2D histogram
  - Lab 색공간 KDE
  - Gradient/contrast feature
  - 구름/태양 주변(near-sun) 영역 따로 분리한 멀티-헤드 시그니처

---

## 4. 시스템 오케스트레이션(필수 구조)

### 4.1 배치/스트리밍 업데이트(전지구 인덱스 생성)

1) **Ingest/Normalize**
- 여러 소스의 시간축을 정렬(예: 10분 단위 time index)
- 타일링(예: 5km 또는 0.05°) 후 통계량 산출
- 품질 플래그/결측 보정 규칙 적용

2) **Feature Store**
- `EarthState`를 point/tile/time 키로 저장
- 재현 가능하도록 원천 데이터 버전/타임스탬프 기록

3) **Signature Compute**
- 1차(빠름): 물리 기반 간이 전방모델(태양 위치 + analytic sky + 단순 지표 팔레트)
- 2차(정확): top-K 후보에 대해서만 고해상도/고정밀(구름/오존/에어로졸) 재추정

4) **Vector Index Update**
- `Signature Vector`를 (tile,time) 키와 함께 인덱싱
- 최신성 기준: *최근 N시간*은 자주 업데이트, 오래된 구간은 저주기 재계산

### 4.2 사용자 쿼리 경로

- 입력: target color(이미지/팔레트/히스토그램) + 시간 조건
- target signature 생성
- Vector search로 Top-K 후보 추출
- 후보에 대해 고정밀 rerank + 설명 생성
- 결과 반환

---

## 5. 모듈 계약(Contracts)

### 5.1 시그니처 계산 커널 (필수 API)
다음 함수(또는 동등한 API)를 제공해야 한다.

- `compute_color_signature(x, earth_state, surface_state, config) -> ColorSignature`

`ColorSignature` 최소 필드:
- `hue_bins: (N,)` (0..1 또는 0..360)
- `sky_hue_hist: (N,)` (sum=1)
- `ground_hue_hist: (N,)` (sum=1)
- `signature: (2N,)`
- `meta`: `sun_elev_deg`, `sza_deg`, `turbidity`, `quality_flags` 등

### 5.2 검색 인덱스 (필수 API)
- `add(keys, vectors)`
- `query(vector, top_k, filters?) -> List[(key, distance)]`
- MVP에서는 in-memory brute force 허용, 추후 ANN/Managed Vector Search로 교체 가능.

### 5.3 데이터 소스 커넥터 (필수 API)
외부 의존을 최소화하기 위해 **인터페이스를 먼저 고정**한다.

- `get_earth_state(time, lat, lon) -> EarthState`
- `get_surface_state(lat, lon) -> SurfaceState`

MVP에서는 모의(Mock)/샘플 데이터로 동작해야 하며, 실서비스 커넥터는 뒤에 붙인다.

---

## 6. 품질/불확실도(필수)

### 6.1 품질 플래그
최소한 다음 플래그를 제공한다.
- `is_night`: 태양고도 < 0
- `low_sun`: 태양고도 < 10°
- `missing_realtime`: 실시간 관측 결측
- `cloudy`: cloud_fraction 상위 임계치 초과

### 6.2 불확실도 점수
- 시그니처에 `uncertainty_score (0..1)`를 포함한다.
- 간단 구현: 결측/구름/낮은 태양고도/고 AOD에서 불확실도 상승.

---

## 7. 성능/운영 목표(SLO) — 초기 기준

- 단일 `(time,lat,lon)` 시그니처 계산: < 200ms (로컬 CPU 기준, MVP)
- 검색 API: < 1.5s (Top-K=50, 로컬 인덱스 기준)
- 전지구 업데이트: 타일 해상도/범위에 따라 조정하되, **최근 2시간**은 가장 최신을 유지

---

## 8. 비기능 요구사항

### 8.1 재현 가능성
- 동일 입력/동일 데이터 버전이면 동일 시그니처를 산출해야 한다(결정적).

### 8.2 설명가능성
- 결과에는 “왜 이 색이 나왔는지”를 요약한다.
  - 예: 태양고도, 구름량, AOD, 오존량, 지표 클래스 비율

### 8.3 보안
- 사용자 PII 저장 금지.
- 좌표/시간 로그는 최소화(집계/샘플링)하고, 필요한 경우 익명화.

---

## 9. 명시적 비목표(Non-goals)

- 1차 버전에서 **완전한 물리 기반 스펙트럴 방사전달(RT) 렌더러** 구현은 목표가 아니다.
- 1차 버전에서 “카메라별 RAW 파이프라인” 정밀 재현은 목표가 아니다.
- 전지구 초고해상도(예: < 500m) 상시 업데이트는 1차 범위 밖.

---

## 10. 레퍼런스 구현 지침 (MVP)

- 하늘색: 태양 위치 + analytic sky(Preetham/Perez 계열) 기반
- 오존/에어로졸/구름: 1차는 간단한 보정/블렌딩, 2차는 잔차 ML 또는 LUT
- 지표: landcover 클래스 비율 + 팔레트(도시/식생/수면/눈 등) 혼합
- 시그니처: Hue histogram (S×V 가중치) + 원형 스무딩

