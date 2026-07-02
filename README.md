# 의류 분석 PoC — 옷 누끼 + 정보 추출

옷 사진 한 장에서 **누끼(배경·인물 제거)** 와 **옷 정보(디자인 속성 · 대표색)** 를 추출하는 PoC.
GPU(NVIDIA) 서버에서 FastAPI로 서빙.

---

## 기능 (API 엔드포인트)

서버 실행 후 `/docs` 에서 대화형 명세 확인 가능.

| 메서드 | 경로 | 설명 | 응답 |
| --- | --- | --- | --- |
| POST | `/segment?category=upper\|lower\|full\|all` | 단일 부위 누끼 | PNG 이미지 |
| POST | `/segment2` | 상/하/전신 누끼 한 번에 | JSON (base64 PNG 3개) |
| POST | `/attrs` | 옷 디자인 속성(무늬/넥라인/소매 등) | JSON |
| POST | `/color` | 대표색 팔레트 | JSON |
| GET | `/` | 헬스체크 | JSON |

모든 이미지 입력은 `multipart/form-data` 의 `file` 필드로 업로드.

### 사용 예시
```bash
# 단일 누끼 (PNG 바로 저장)
curl -X POST "localhost:8000/segment?category=upper" -F "file=@sample.jpg" --output upper.png

# 속성 추출
curl -X POST "localhost:8000/attrs" -F "file=@sample.jpg"

# 대표색 (누끼된 이미지 권장)
curl -X POST "localhost:8000/color" -F "file=@cutout.png"
```

---

## 실행

### A. Docker (권장)

```bash
# NVIDIA GPU 서버 (x86_64 + nvidia-container-toolkit)
docker pull rabbitjun/clothing-api
docker run --rm --gpus all -p 8000:8000 rabbitjun/clothing-api

# GPU 없는 환경 (CPU 폴백, 느림)
docker run --rm -p 8000:8000 rabbitjun/clothing-api
```
→ `http://<서버>:8000/docs` 접속.

> 직접 빌드: `docker build -t clothing-api . && docker run --rm --gpus all -p 8000:8000 clothing-api`
> HF 모델 캐시 재사용(다운로드 생략): `-v ~/.cache/huggingface:/root/.cache/huggingface`

### B. 로컬 (개발용)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Apple Silicon(Mac)은 MPS로 동작. 미지원 연산 대비: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- 모델은 최초 실행 시 HuggingFace에서 자동 다운로드 → 캐시(`~/.cache/huggingface`).

---

## 모델 (1차 = 성능 기준 선정)

| 용도 | 채택 모델 | 근거 |
| --- | --- | --- |
| 누끼 | `mattmdjaga/segformer_b2_clothes` | 직접 마스크 중 최속·최경량 |
| 속성 | `patrickjohncyh/fashion-clip` (제로샷) | 무늬/넥라인/소매 등 |
| 색상 | K-means (`colors.py`) | 고전 CV, CPU, 무비용 |

### 벤치마크 요약 (RTX 4090, 입력 5장 평균, VRAM = nvidia-smi 프로세스 전체)

| 모델/파이프라인 | 속도 | VRAM | 비고 |
| --- | --- | --- | --- |
| yolos-fashionpedia | 12.5 ms | 0.77 GB | 검출(구조부위 포함) |
| segformer_b2_clothes | 18.5 ms | 1.06 GB | **직접마스크 누끼** |
| yolos + SAM2 | 44.9 ms | 1.88 GB | 인스턴스 누끼 |
| rembg (u2net_cloth_seg) | 53.2 ms | 2.40 GB | 상/하/전신, onnxruntime |
| FashionCLIP | 81.2 ms | 1.09 GB | 속성 |
| grounding-dino | 81.4 ms | 3.81 GB | 텍스트 검출 |
| grounding-dino + SAM2 | 115.1 ms | 3.93 GB | 텍스트 누끼 |

> 벤치마크 재현: `python bench.py all` (모델별 별도 프로세스로 격리 측정)

---

## 프로젝트 구조

```
main.py            FastAPI 앱 (/segment, /segment2, /attrs, /color)
seg1.py            segformer 누끼          detect.py         yolos 검출
sam_cutout.py      yolos + SAM2           ground_detect.py  grounding-dino 검출
ground_sam.py      grounding-dino + SAM2  seg_rembg.py      rembg
pattern.py         FashionCLIP 속성        colors.py         K-means 대표색
device_check.py    디바이스 선택(cuda/mps/cpu)

bench.py           속도·VRAM 벤치마크
generate.py        모델별 부위 누끼 이미지 생성 (품질 비교용)
extract_info.py    이미지별 속성+색상 추출 → output/info.json
decode.py          /segment2 base64 응답 → PNG 디코드 헬퍼

Dockerfile · requirements.txt · .dockerignore
input/             샘플 이미지            output/           생성 결과(gitignore)
```

---

## 유틸 스크립트

```bash
python bench.py all        # 전체 모델 속도·VRAM 벤치마크
python generate.py         # 모델별 상/하/전신 누끼 이미지 → output/compare/
python extract_info.py     # 이미지별 속성+대표색 → output/info.json
```

---

## 참고 / 주의

- **rembg(onnxruntime)는 GPU 세팅이 까다로움**: `onnxruntime-gpu` 단독 설치 + CUDA 라이브러리 경로 필요.
  ```bash
  export LD_LIBRARY_PATH="$(find $PWD/.venv -type d -path '*/nvidia/*/lib' | tr '\n' ':')$LD_LIBRARY_PATH"
  ```
- rembg를 직접 실행(seg_rembg.py, generate.py 등)할 때만 위 LD_LIBRARY_PATH 설정 필요.
  서빙 API(main.py)는 누끼에 segformer를 쓰므로 rembg·이 설정과 무관.
- **아키텍처**: 배포 이미지는 `linux/amd64`. ARM(Apple Silicon)에서 실행 시 `--platform linux/amd64`(에뮬, 느림).

---

## 로드맵

- [x] 모델 벤치마크 + 실측
- [x] 1차 FastAPI (성능 기준) + Docker
- [ ] SCHP 
- [ ]claude,gemini
