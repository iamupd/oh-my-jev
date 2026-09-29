<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.png">
  <img src="docs/images/logo.png" alt="oh-my-jev 로고" width="120">
</picture>

# oh-my-jev

**판단 모델을 만들고, 언제 믿어도 되는지 측정함.**

오픈 *System One* 판단 모델을 서빙·벤치마크·학습·비교하는 도구임. 판단 모델은 정해진 질문에 자유 문장 대신 보정된 확률로 답하는 작은 모델임.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Release](https://img.shields.io/badge/release-v0.1.0-2ea44f)](https://github.com/iamupd/oh-my-jev/releases)
[![TypeSafe compatible](https://img.shields.io/badge/API-TypeSafe%20%2Fv1%2Fsystemone-6f42c1)](#typesafe-sdk로-연결)

[빠른 시작](#빠른-시작) · [명령](#명령) · [학습](#판단-모델-직접-학습) · [알려진 제한](#알려진-제한) · [English](README.md)

<img src="docs/images/bench.png" alt="omj bench 출력: 지표, 태그별 분해, 신뢰도 표, 요약" width="760">

</div>

## 왜 필요한가

에이전트는 작은 판단을 많이 함: *이 문의는 어느 팀으로 보낼지, 이 답변은 근거가 있는지, 사람이 확인해야 하는지.*

- 생성된 문장에는 기준으로 삼을 값이 없음
- 판단 모델은 선택지마다 확률을 주므로 0.9 이상이면 자동 처리하고 그 아래는 사람에게 넘길 수 있음. 단, 그 확률이 정직해야 함
- oh-my-jev는 이 과정을 위한 도구임: TypeSafe 호환 엔드포인트로 서빙하고, 정확도와 **보정**을 함께 측정하고, 직접 학습하고, 기준 모델과 비교함

## 주요 특징

- 🚀 **1회 순전파 판독.** 로컬 모델은 순전파 1회의 선택지 레이블 로짓으로 답함. 문장 생성 없이 판단 1건당 prefill 1회
- 🎯 **정확도만이 아니라 보정까지.** ECE, Brier, NLL, 위험 5% 기준 커버리지, 신뢰도 표, 95% 신뢰구간, 온도 보정
- 🔌 **그대로 연결되는 엔드포인트.** `omj serve`는 TypeSafe의 `/v1/systemone` 형식이라 공식 SDK는 `base_url`만 바꾸면 됨
- 🧪 **직접 학습.** 레이블 한정 교차엔트로피와 Brier 항을 쓰는 LoRA / QLoRA 레시피. 작은 GPU용 4-bit 로딩 지원
- ⚖️ **나란히 비교.** 터미널에서 보고서를 비교하거나, 웹 플레이그라운드에서 한 상황을 두 모델에 보냄 (영어 / 한국어)
- 🧰 **설치 직후 바로 동작.** mock 백엔드는 GPU와 키가 필요 없어 모델을 받기 전에도 모든 명령을 실행할 수 있음

## 빠른 시작

[uv](https://docs.astral.sh/uv/)와 Python 3.11 이상이 필요함.

```bash
git clone https://github.com/iamupd/oh-my-jev.git
cd oh-my-jev
uv sync --extra dev
uv run omj init --yes                 # GPU와 키를 감지해 ~/.omj/config.toml 생성
uv run omj bench --suite omj-smoke    # init이 고른 백엔드로 30문항 실행
uv run omj serve                      # http://127.0.0.1:8799, Ctrl+C로 종료
```

두 번째 터미널에서 요청을 보냄:

```bash
curl -s http://127.0.0.1:8799/v1/systemone -H "Content-Type: application/json" -d '{"model": "jev-latest", "state": "Customer: I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payment problems", "shipping": "delivery problems"}}}}'
```

<details>
<summary>Windows PowerShell</summary>

PowerShell에서는 `curl`이 `Invoke-WebRequest`의 별칭이므로 같은 요청을 다음처럼 보냄:

```powershell
$body = '{"model": "jev-latest", "state": "Customer: I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payment problems", "shipping": "delivery problems"}}}}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8799/v1/systemone -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

</details>

<details>
<summary><code>omj init</code>의 백엔드 선택 규칙</summary>

- 6~24 GB CUDA GPU이고 `semif` extra가 설치돼 있으면 로컬 Qwen3.5 모델 ([로컬 모델 실행](#로컬-모델-실행) 참고)
- `semif` extra가 없으면 `JEV_KEY` 또는 `OPENROUTER_KEY`가 있을 때 원격 백엔드, 없을 때 mock 백엔드. 로컬 모델 활성화 방법을 함께 출력함
- 직접 지정: `uv run omj init --yes --backend mock` (또는 `typesafe`, `semif`)
- `omj bench`는 `--backend NAME`이나 `--endpoint URL`을 주지 않으면 설정 파일의 백엔드를 씀

</details>

## 명령

| 명령 | 기능 |
|---|---|
| `omj init` | 하드웨어·키 감지, 동작 확인, `~/.omj/config.toml` 생성 |
| `omj serve` | 로컬 모델·원격 API·mock 위에 TypeSafe 호환 `/v1/systemone` 엔드포인트 제공 |
| `omj bench` | 정확도, 보정, 유형별·태그별 분해, 신뢰도 표. `report.md`와 `report.json`으로 저장 |
| `omj compare` | 보고서 2개 이상을 나란히 놓고 첫 번째 대비 차이 표시 |
| `omj train` | TOML 레시피로 LoRA / QLoRA 미세조정 |
| `omj ui` | 모델 1~2개를 비교하는 로컬 웹 플레이그라운드, 정책 임계값 적용 |

각 명령은 `uv run omj …`로 실행함. `.venv`를 활성화하면 `omj …`만으로도 실행할 수 있음.

| 백엔드 | 동작 | 필요한 것 |
|---|---|---|
| `semif` | 오픈 가중치를 직접 로드하고 선택지 로짓을 판독. LoRA adapter 적용 가능 | CUDA GPU, `uv sync --extra semif` |
| `typesafe` | 호스팅된 Jev API를 직접 또는 OpenRouter를 거쳐 호출 | `JEV_KEY` 또는 `OPENROUTER_KEY` |
| `kev` | 실행 중인 [Kev](https://github.com/jaredpalmer/kev) 서버 호출 | Kev 서버 |
| `mock` | 키워드 겹침으로 점수 계산. 모델이 아니라 테스트용 | 없음 |

## 기준 모델과 비교

```bash
uv run omj compare jev/report.json coco-v13/report.json coco-v14/report.json --labels jev-1.13,coco-v13-nf4,coco-v14-nf4 --out runs/cmp
```

<div align="center">
<img src="docs/images/compare.png" alt="omj compare 출력: 보고서 3개의 지표와 색으로 구분한 차이" width="760">
</div>

- 차이는 지표의 방향에 맞춰 색으로 표시함(초록이 더 좋음). 행마다 가장 좋은 값은 굵게 표시함
- 위 수치는 2026년 9월 JevBench 공개 문항 231개로 측정한 값임

## 로컬 모델 실행

```bash
uv sync --extra semif --extra dev    # torch, transformers, peft 설치(수 GB)
uv run omj init --yes                # 6~24 GB CUDA GPU에서 Qwen3.5 기반 모델 다운로드(2B 약 4.5 GB)
uv run omj serve
```

- `--no-download`를 주면 가중치를 받지 않고 설정만 저장함
- 8 GB GPU에서는 `config.toml`의 `[backend]`에 `quant = "nf4"`를 넣어 4B 모델을 4-bit로 로드함

## 판단 모델 직접 학습

```bash
uv sync --extra semif --extra dev
uv run omj train --recipe example-intent-en    # Qwen3.5-0.8B 기반 조정 전 소형 예제
uv run omj bench --adapter ~/.omj/adapters/example-intent-en/<run_id>/best --suite omj-holdout
```

- 레시피에는 데이터 출처(`[[data.mix]]`), LoRA 설정, 학습 설정이 들어감. 내장 예제는 출발점이며, 데이터 출처를 자신의 판단 데이터로 바꿔 쓰면 됨
- `--adapter`를 주면 `bench`와 `serve`는 adapter를 학습한 기반 모델을 로드함
- 체크포인트는 `omj-holdout`(BoolQ, ARC-Challenge, CommonsenseQA, SVAMP에서 뽑은 300문항, 내장 학습 데이터 미사용)으로 고르는 것을 권함. 봉인된 시험 세트가 아니라 공개 개발용 세트임. 공개 벤치마크 수치는 보고만 하고 선택 기준으로 쓰지 않음

## TypeSafe SDK로 연결

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8799")
```

웹 플레이그라운드(`uv run omj ui --target a=mock --target b=mock`)는 영어로 열림. 주소에 `?lang=ko`를 붙이거나 **EN / 한국어** 버튼으로 한국어로 바꿀 수 있음.

## 알려진 제한

- Windows 11과 NVIDIA GPU에서 테스트했음. Linux에서도 동작할 것으로 예상함. 로컬 모델은 CUDA가 필요해 macOS에서는 지원하지 않음
- `omj train`은 보정 온도를 모든 문항을 `choice`로 보고 학습함. 혼합 레시피의 yes/no·`score` 문항은 같은 온도를 공유함. 내장 예제는 `choice`만 사용함
- MASSIVE 스위트와 학습은 처음 사용할 때 데이터를 내려받음
- `omj compare`는 완성된 보고서를 비교함. 모델을 실행하거나 차이의 유의성을 검정하지 않음

## 감사의 말

oh-my-jev는 아래 프로젝트의 아이디어와 인터페이스를 바탕으로 만들었으며, 코드를 복사하지는 않았음:
[TypeSafe](https://docs.typesafe.ai) (System One 통신 규약과 SDK),
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (1회 순전파 선택지 로짓 판독),
[Kev](https://github.com/jaredpalmer/kev)와 [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) (LoRA로 학습한 공개 판단 모델),
[JevBench](https://github.com/fstandhartinger/jevbench) (공개 문항과 채점 방식).

## 라이선스

코드: [Apache-2.0](LICENSE). 번들 평가 데이터는 각자의 라이선스를 따름([suites/NOTICE.md](suites/NOTICE.md)). 의존성과 데이터셋 목록은 [THIRD-PARTY.md](THIRD-PARTY.md) 참고.

> TypeSafe AI와 제휴하거나 보증받은 프로젝트가 아님. "Jev"와 "System One"은 TypeSafe AI가 사용하는 명칭이며, 이 프로젝트는 호환 인터페이스를 구현한 독립 프로젝트임.
