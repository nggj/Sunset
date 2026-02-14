# AGENTS.md

이 문서는 Codex(또는 자동화 에이전트)가 **반복적으로 일관된 품질**로 구현하기 위한 규칙을 고정합니다.

## 1) 개발 원칙

### 1.1 Single Source of Truth
- 제품/계약/용어/오케스트레이션 기준은 `docs/spec.md`가 **최우선**입니다.
- 코드가 spec와 충돌하면 **spec에 맞게 코드 변경**이 원칙입니다.

### 1.2 인터페이스 우선
- 외부 시스템(위성/기상/벡터 검색/클라우드) 연동은 **인터페이스를 먼저 고정**하고,
  - MVP에서는 **Mock/샘플 구현**으로 end-to-end를 통과시킵니다.
  - 실제 커넥터는 후속 태스크에서 교체/추가합니다.

### 1.3 결정성(Determinism)
- 동일 입력 + 동일 데이터 버전이면 결과가 동일해야 합니다.
- 난수 사용 시 seed를 고정하고, 테스트에서 재현 가능해야 합니다.

---

## 2) 코드 스타일/품질 규칙

### 2.1 언어/버전
- Python 3.11+ 기준.

### 2.2 포맷/린트/타입
- 포맷/린트: `ruff`
- 테스트: `pytest`
- 타입체크: `mypy` (또는 `pyright`, 둘 중 하나로 통일)

> 도입 도구는 태스크 1에서 pyproject/설정으로 고정합니다.

### 2.3 네이밍/구조
- 패키지 루트: `src/skycolor_locator/`
- 모듈은 기능 기준으로 분리:
  - `astro/` (태양 위치)
  - `sky/` (analytic sky)
  - `signature/` (히스토그램/시그니처)
  - `ingest/` (EarthState 커넥터 인터페이스)
  - `index/` (Vector index)
  - `api/` (FastAPI)

### 2.4 문서화
- public 함수에는 docstring 필수.
- spec 변경이 필요하면 PR/커밋에서 **spec부터 업데이트**.

---

## 3) 실행 명령(로컬 기준)

> 아래 명령은 repo에 `pyproject.toml`과 dev deps가 준비되면 동작해야 합니다.

### 3.1 설치
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 3.2 품질
```bash
ruff check .
ruff format .
pytest -q
mypy src
```

### 3.3 실행(MVP)
- CLI:
```bash
python -m skycolor_locator --help
```
- API 서버(구현 시):
```bash
uvicorn skycolor_locator.api.app:app --reload --port 8000
```

---

## 4) 금지사항(반드시 준수)

### 4.1 비밀정보/키
- API 키/토큰/자격증명/개인정보를 repo에 커밋 금지.
- `.env`는 예시 템플릿만 제공하고 실제 값은 넣지 않습니다.

### 4.2 네트워크/외부 의존
- 테스트는 **네트워크 없이** 통과해야 합니다.
- 외부 API 호출은 기본적으로 금지. 필요하면:
  - 인터페이스 뒤로 숨기고
  - 기본 구현은 Mock
  - 실제 구현은 opt-in(환경변수로 enable)

### 4.3 무거운 의존성
- ANN 라이브러리, 대형 ML 프레임워크 등 무거운 의존성은 **MVP에서 최소화**.
- 꼭 필요하면 “왜 필요한지”를 태스크 프롬프트/커밋 메시지에 명확히 남길 것.

### 4.4 스펙 무시/암묵적 변경
- `docs/spec.md`의 계약(입력/출력 스키마, 필수 API)을 조용히 바꾸지 말 것.
- 변경이 필요하면 spec을 먼저 수정하고, 변경 이유/마이그레이션을 함께 기록.

---

## 5) 테스트/검증 규칙

- 단위 테스트는 다음을 포함:
  - 태양 위치(기본 sanity)
  - 시그니처 벡터 shape/정규화(sum=1)
  - 결정성(동일 입력 동일 출력)

- 통합 테스트(MVP):
  - Mock EarthState + Mock SurfaceState로
  - `/signature` 또는 CLI 한 번 호출 시
  - signature 생성 → index add → query까지 end-to-end 1회 통과

---

## 6) 작업 방식(에이전트용)

- 각 태스크는 **작게** 완료하고, 테스트/린트까지 통과시키고, 다음 태스크로 이동.
- TODO/주석으로 미뤄둔 부분이 있으면, 반드시:
  - 남은 작업 범위
  - 스펙 영향
  - 다음 태스크에서 해결할 계획
  을 명확히 적어둘 것.

