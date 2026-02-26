# LG Aimers 8기 : 모델 경량화 온라인 해커톤

> **DACON 해커톤** | 2026.02 | 최종 순위: **239 / 628** | 최종 스코어: **0.59878**

## 대회 개요

| 항목 | 내용 |
|------|------|
| 주최 | LG AI Research (LG Aimers) |
| 플랫폼 | DACON |
| 주제 | LLM 경량화 (Large Language Model Compression) |
| 베이스 모델 | EXAONE-4.0-1.2B (LG AI Research) |
| 평가 환경 | Ubuntu 22.04, 4 vCPU, 16GB RAM, NVIDIA A10G (22.4GB VRAM), CUDA 12.8 |
| 최종 순위 | 239 / 628 |
| 최종 스코어 | 0.59878 |

## 평가 지표

$$\text{Score} = \max\left(0.5 \times \text{Perf\_norm} + 0.5 \times \text{SpeedNorm}, \ 0\right)$$

- **Perf_norm**: 경량화 후 모델의 성능을 기준 모델(EXAONE-4.0-1.2B) 대비 정규화한 값
- **SpeedNorm**: 기준 모델 대비 추론 속도 향상 비율 정규화 값
- 성능과 속도를 동시에 고려하는 복합 지표

## 사용 기술 스택

- **경량화 기법**: GPTQ (W4A16 quantization), AWQ, Structural Pruning, LoRA
- **주요 라이브러리**: `llmcompressor`, `transformers`, `peft`, `torch`
- **캘리브레이션 데이터**: MANTA-1M, KMMLU
- **추론 서버**: vLLM (`gpu_memory_utilization=0.85`, `batch_size=auto`)

---

## 시도한 방법 정리

### 최종 제출 (Try_014) — Score: 0.59878

| 설정 | 값 |
|------|-----|
| 경량화 기법 | GPTQ W4A16 |
| 캘리브레이션 데이터 | MANTA-1M 128샘플 + KMMLU 32샘플 (총 160샘플) |
| Group size | 128 |
| Max seq length | 512 |
| Ignore layers | embed_tokens, lm_head (tied weights) |
| 모델 크기 | ~814 MB |

두 가지 도메인의 데이터를 혼합하여 한국어 MCQ(Multiple Choice Question)와 일반 대화 패턴을 모두 반영한 Hessian을 계산해 GPTQ 양자화 품질을 높이는 전략.

---

### 전체 실험 흐름

```
Try_000  →  Try_014  →  Try_017
[Baseline]  [2-source Calib]  [LoRA + GPTQ]
```

#### ✅ Try_000 — GPTQ 기본 베이스라인

- **기법**: GPTQ W4A16
- **캘리브 데이터**: MANTA-1M 128샘플
- **결과**: Phase2 = 0.5914 (기준점)

**핵심 설정:**
```yaml
GPTQModifier:
  targets: [Linear]
  ignore: [embed_tokens, lm_head]
  scheme: W4A16
  block_size: 128
  dampening_frac: 0.01
```

---

#### ✅ Try_014 — 2-source 캘리브레이션 (최종 제출)

- **기법**: GPTQ W4A16
- **캘리브 데이터**: MANTA-1M 128샘플 + **KMMLU 32샘플** (한국어 벤치마크 추가)
- **주요 버그 수정**:
  - KMMLU의 "all" config 부재 → subject별 로드로 변경
  - `model.tie_weights()` 미호출 시 lm_head 중복 저장 (814MB → 1.4GB) 문제 수정
  - submit.zip 내부 경로 구조 (`model/` 루트) 수정
- **결과**: **0.59878** (최종 제출)

**아이디어**: GPTQ는 캘리브레이션 데이터 기반으로 Hessian을 계산하므로, 평가 태스크와 유사한 분포의 데이터를 추가하면 양자화 오차를 줄일 수 있다는 가설.

---

#### 🔬 Try_017 — Pre-GPTQ LoRA 정렬 + GPTQ (QAT-inspired)

- **기법**: LoRA fine-tuning → merge → GPTQ W4A16
- **파이프라인**:
  1. FP16 모델 로드 (bfloat16)
  2. LoRA 파인튜닝 (200 steps, r=16, α=32)
     - Target: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
     - Data: MANTA 128 + KMMLU 32
  3. LoRA 가중치 병합 (`merge_and_unload`)
  4. GPTQ W4A16 적용
  5. `tie_weights()` → save → ZIP
- **검증**: unique_ratio ≥ 0.3 (반복 생성 방지)

**아이디어**: GPTQ가 Hessian을 계산하기 전에 LoRA로 모델 가중치 분포를 캘리브레이션 데이터 분포에 정렬(align)시키면, Hessian 추정이 더 정확해져 양자화 오차가 감소한다는 가설 (QAT에서 영감).

---

### 폐기된 실험들 (trash/)

| 실험 | 시도 내용 | 폐기 이유 |
|------|-----------|-----------|
| AWQ | Per-channel scaling 기반 대안 양자화 | GPTQ 대비 성능 열위 |
| Pruning + LoRA | 구조적 프루닝 후 LoRA | 1.2B 규모에서 과도한 성능 손실 |
| Seq length 1024 | 더 긴 캘리브 시퀀스 | RAM 부족 (25.8GB 필요) |
| Data shuffle | 캘리브 데이터 순서 변경 | 유의미한 차이 없음 |
| Data filtering | 캘리브 데이터 필터링 | 유의미한 차이 없음 |
| Colab 실행 | Google Colab 환경 | lm_head 중복 버그 재현 |

---

## 프로젝트 구조

```
LG_aimers/
├── notebooks/
│   ├── Try_000.ipynb               # GPTQ 베이스라인
│   ├── Try_014_gptq_bench.ipynb    # 최종 제출 (2-source 캘리브)
│   └── Try_017_lora_gptq.ipynb     # LoRA + GPTQ 실험
└── README.md
```

> `submit/` (제출 zip 파일들, ~30GB)과 `model/` (학습된 가중치)은 용량 문제로 제외.

---

## 주요 인사이트 및 배운 점

1. **tied weights 처리**: EXAONE처럼 embed_tokens와 lm_head가 가중치를 공유하는 모델은 save 전 `model.tie_weights()`를 반드시 호출해야 중복 저장을 방지함.

2. **캘리브레이션 데이터의 중요성**: GPTQ에서 캘리브 데이터 선택은 최종 성능에 직결됨. 평가 도메인과 유사한 데이터(KMMLU) 추가가 효과적.

3. **AWQ vs GPTQ**: 이 태스크에서는 GPTQ W4A16이 AWQ보다 일관되게 우수한 결과를 보임.

4. **하드웨어 제약**: Mac MPS 환경에서는 GPTQ의 oneshot()이 CPU fallback으로 동작 → 개발/테스트 시 주의.

5. **모델 파일 크기**: submit.zip 내 lm_head 중복 시 814MB → 1.4GB로 증가. 채점 환경 메모리 한계를 넘길 수 있어 반드시 확인 필요.

---

## 참고 자료

- [DACON 대회 페이지](https://dacon.io/competitions/official/236673/overview/description)
- [EXAONE-4.0-1.2B (HuggingFace)](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [KMMLU Dataset](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)
- [MANTA-1M Dataset](https://huggingface.co/datasets/maywell/MANTA-1M)
