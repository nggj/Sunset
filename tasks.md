# Implementation Task Prompts (for Codex)

> 각 태스크는 단독으로 Codex에 붙여넣어 실행할 수 있는 **프롬프트**입니다.
> 모든 태스크는 `docs/spec.md`와 `AGENTS.md`를 준수해야 합니다.

---

## Task 1 — Repo Scaffold + Tooling Baseline

**Goal**: Python 패키지 스캐폴딩과 개발 도구(ruff/pytest/mypy)를 세팅하고, 최소 실행 가능한 CLI 엔트리포인트를 만든다.

**Instructions**:
1. `pyproject.toml`을 추가하고 패키지 이름을 `skycolor-locator`(import는 `skycolor_locator`)로 구성.
2. `src/skycolor_locator/__init__.py` 및 `src/skycolor_locator/__main__.py` 생성.
3. `python -m skycolor_locator --help`가 동작하도록 argparse 기반 CLI 추가.
4. dev dependencies에 ruff/pytest/mypy 추가.
5. `tests/test_smoke.py`에서 CLI import smoke test 추가.

**Acceptance Criteria**:
- `ruff check .` 통과
- `pytest -q` 통과
- `python -m skycolor_locator --help`가 정상 출력

---

## Task 2 — Data Contracts (EarthState / SurfaceState / ColorSignature)

**Goal**: spec에 정의된 데이터 계약을 코드로 고정하고, 검증/직렬화를 제공한다.

**Instructions**:
1. `src/skycolor_locator/contracts.py` 생성.
2. 아래 dataclass(또는 pydantic model) 정의:
   - `AtmosphereState`
   - `SurfaceClass`, `SurfaceState`
   - `ColorSignature` (hue_bins, sky_hue_hist, ground_hue_hist, signature, meta, uncertainty_score, quality_flags)
3. numpy array 직렬화(리스트 변환) 유틸 제공.
4. 최소한의 validation:
   - hist는 음수 금지
   - hist sum ~= 1
   - signature 길이=2N

**Acceptance Criteria**:
- 단위 테스트에서 정상 생성/검증
- `ColorSignature.to_dict()`로 JSON 직렬화 가능

---

## Task 3 — Solar Position Module + Tests

**Goal**: `(time_utc, lat, lon) -> (sza_deg, saz_deg, sun_elev_deg)` 계산을 모듈화하고 테스트한다.

**Instructions**:
1. `src/skycolor_locator/astro/solar.py` 생성.
2. `solar_position(dt, lat_deg, lon_deg)` 구현.
3. 테스트:
   - 위도 0, 경도 0에서 춘분 근처 정오(UTC 기준은 근사)처럼 기본 sanity를 체크하는 범위 테스트
   - `sun_elev_deg = 90 - sza_deg` 일관성
   - 결정성

**Acceptance Criteria**:
- `pytest -q` 통과

---

## Task 4 — Analytic Sky Model (Preetham/Perez) + Ozone/Cloud Heuristics

**Goal**: analytic sky로 하늘 돔 RGB를 생성하는 baseline을 구현한다.

**Instructions**:
1. `src/skycolor_locator/sky/analytic.py` 생성.
2. `render_sky_rgb(dt, lat, lon, atmos, n_az, n_el) -> (rgb, meta)` 구현.
   - 태양 위치는 Task 3 모듈 사용
   - turbidity는 AOD/visibility 기반 heuristic
   - Perez 분포 함수 기반
   - 오존/구름은 1차는 heuristic blend/correction
3. 테스트:
   - shape: (n_el, n_az, 3)
   - 값 범위: 0..1
   - cloud_fraction 증가 시 채도 감소/회색화 경향(정확한 수치 X, 단조성 정도)

**Acceptance Criteria**:
- `pytest -q` 통과

---

## Task 5 — Color Signature Computation (Sky/Ground) + Hist Curves

**Goal**: spec의 핵심 커널 `compute_color_signature`를 구현한다.

**Instructions**:
1. `src/skycolor_locator/signature/core.py` 생성.
2. 다음 기능 구현:
   - `srgb_to_hsv`
   - `hue_histogram(rgb, bins, weight_mode="sv")`
   - `smooth_circular`
   - `compute_color_signature(dt, lat, lon, atmos, surface, config) -> ColorSignature`
3. ground는 팔레트 혼합 + 간이 조도 + haze(지평선색 blend)
4. meta에 태양고도/품질플래그/uncertainty_score 포함.
5. 테스트:
   - sky/ground hist 각각 sum=1
   - signature dim=2*bins
   - 동일 입력 동일 출력

**Acceptance Criteria**:
- `pytest -q` 통과

---

## Task 6 — Ingest Interfaces (Connectors) + Mock Implementations

**Goal**: 외부 데이터 소스(Earth Engine/기상)를 직접 붙이기 전에, 인터페이스를 고정하고 Mock으로 end-to-end가 가능하게 한다.

**Instructions**:
1. `src/skycolor_locator/ingest/interfaces.py`에 Protocol(또는 ABC)로 정의:
   - `EarthStateProvider.get_atmosphere_state(dt, lat, lon) -> AtmosphereState`
   - `SurfaceProvider.get_surface_state(lat, lon) -> SurfaceState`
2. `src/skycolor_locator/ingest/mock_providers.py` 구현:
   - 위도/경도에 따라 landcover 팔레트를 대충 다르게 반환(예: 해양/내륙/도시)
   - 시간에 따라 구름량이 약간 변하는 결정적 함수
3. `src/skycolor_locator/ingest/cache.py` 간단한 LRU 캐시(옵션)
4. 통합 테스트에서 mock provider로 signature 1개 생성 가능해야 함.

**Acceptance Criteria**:
- 네트워크 없이 `pytest -q` 통과

---

## Task 7 — Vector Index (Local MVP) + Similarity Metrics

**Goal**: 시그니처 벡터를 저장하고 검색하는 로컬 인덱스를 구현한다.

**Instructions**:
1. `src/skycolor_locator/index/base.py`에 인터페이스 정의:
   - `add(keys: list[str], vectors: np.ndarray)`
   - `query(vector: np.ndarray, top_k: int) -> list[tuple[str, float]]`
2. `src/skycolor_locator/index/bruteforce.py` 구현(내적/코사인 거리)
3. `src/skycolor_locator/index/metrics.py`에 cosine_distance, emd_1d 추가
4. 테스트:
   - 3개 벡터 넣고 가장 가까운 결과가 기대와 일치

**Acceptance Criteria**:
- `pytest -q` 통과

---

## Task 8 — API Server (FastAPI) for /signature and /search

**Goal**: signature 생성과 색감 검색을 제공하는 API를 만든다.

**Instructions**:
1. `fastapi`/`uvicorn`을 optional dependency로 추가(또는 기본 deps로 포함).
2. `src/skycolor_locator/api/app.py` 생성.
3. 엔드포인트:
   - `POST /signature` body: {time_utc, lat, lon} -> ColorSignature JSON
   - `POST /search` body: {target_signature, time_utc(optional), top_k} -> candidates
4. 내부적으로 mock providers + bruteforce index로 MVP 동작.
5. OpenAPI 스키마가 spec의 계약을 반영.

**Acceptance Criteria**:
- `uvicorn ...`으로 서버 실행 가능
- `pytest -q` 통과 (API는 TestClient로 최소 1개 테스트)

---

## Task 9 — Batch Orchestrator (Local) + Demo Script

**Goal**: 최근 시간 구간에 대해 타일 그리드 시그니처를 생성하고 인덱싱한 뒤, target으로 검색하는 데모를 만든다.

**Instructions**:
1. `src/skycolor_locator/orchestrate/batch.py` 생성.
2. 기능:
   - lat/lon 범위 + step(예: 5°) 그리드 생성
   - 각 포인트에 대해 signature 계산
   - index에 add
3. `scripts/demo_local_search.py` 추가:
   - 지금(UTC) 기준 signature index 생성
   - 임의 포인트의 signature를 target으로 하여 top_k 검색
   - 결과를 콘솔에 pretty print

**Acceptance Criteria**:
- `python scripts/demo_local_search.py` 실행 시 에러 없이 결과 출력
- `pytest -q` 통과

