# 의류 분석 PoC — 옷 누끼 + 정보 추출

옷 사진 한 장에서 **누끼(배경·인물 제거)** 와 **옷 정보(디자인 속성 · 대표색)** 를 추출하는 PoC.
GPU(NVIDIA) 서버에서 FastAPI로 서빙.

---

## 기능 (API 엔드포인트)

서버 실행 후 `/docs` 에서 대화형 명세 확인 가능.

| 메서드 | 경로 | 설명 | 응답 |
| --- | --- | --- | --- |
| POST | `/analyze` | **누끼+속성+색상 통합** (부위별) | JSON — `image`=누끼 **URL** |
| POST | `/analyze_path` | `/analyze`와 동일, 이미지를 **파일 경로**로 | JSON — `image`=`static/…` 경로 |
| POST | `/segment?category=upper\|lower\|full\|all` | 단일 부위 누끼 | PNG 이미지 |
| POST | `/segment2` | 상/하/전신 누끼 한 번에 | JSON (base64 PNG 3개) |
| POST | `/attrs` | 옷 디자인 속성(무늬/넥라인/소매 등) | JSON |
| POST | `/color` | 대표색 팔레트 | JSON |
| GET | `/` | 헬스체크 | JSON |

모든 이미지 입력은 `multipart/form-data` 의 `file` 필드로 업로드.

### 사용 예시
```bash
# 통합 분석 (누끼 URL + 속성 + 색상) — 부위별 리스트로 반환
curl -X POST "localhost:8000/analyze" -F "file=@sample.jpg"

# 단일 누끼 (PNG 바로 저장)
curl -X POST "localhost:8000/segment?category=upper" -F "file=@sample.jpg" --output upper.png

# 속성 추출
curl -X POST "localhost:8000/attrs" -F "file=@sample.jpg"

# 대표색 (누끼된 이미지 권장)
curl -X POST "localhost:8000/color" -F "file=@cutout.png"
```

**`/analyze` 응답 예** — 사진에 있는 부위(상의/하의/전신)마다 하나씩:
```json
[
  {"image": "http://<서버>:8000/static/ab12_upper.png", "type": "upper",
   "meta_data": {"pattern": "graphic print", "neckline": "v neck", "sleeve": "sleeveless"},
   "color": "black"}
]
```
> **`image` 반환 방식 2가지** — 서버/백엔드 구성에 맞춰 선택:
> - `/analyze` → **URL** (`http://…/static/…png`). 서버·네트워크가 다를 때. GET 하면 이미지.
> - `/analyze_path` → **파일 경로** (`static/…png`). 백엔드가 **같은 호스트에서 static 볼륨을 공유**해 파일을 직접 읽을 때 ([아래 Docker 볼륨 공유](#실행) 참고).

---

## 실행

### 빠른 시작 (팀원용) 🚀

레포를 받고 바로 실행. **이미지는 `docker compose up` 시 Docker Hub에서 자동으로 받아옵니다** (별도 `docker pull`·빌드 불필요).

```bash
# 1. 레포 clone (설정 파일 받기)
git clone https://github.com/ASM-17-NS/Clothing_background_removal_and_information_extraction.git
cd Clothing_background_removal_and_information_extraction   # 폴더명은 clone 결과에 맞춰

# 2. 실행
#   GPU 없는 환경 (CPU)     → docker-compose.yml 사용
docker compose up
#   GPU 서버 (NVIDIA)       → docker-compose.gpu.yml 하나만 사용
docker compose -f docker-compose.gpu.yml up

# 3. 접속:  http://localhost:8000/docs  → /analyze 또는 /analyze_path 사용
```

- 처음 실행 시 이미지(약 3GB)와 모델을 받느라 시간이 걸리고, 이후엔 캐시되어 빠릅니다.
- 누끼 결과는 호스트의 `./static/` 폴더에 저장됩니다 (아래 볼륨 설명 참고).
- 이미지만 따로 받고 싶으면: `docker compose pull` (yml이 가리키는 이미지를 Docker Hub에서 받음).

### A. docker compose (권장)

포트·볼륨 설정이 `docker-compose.yml`에 들어있어 **한 줄로 실행**됩니다.

```bash
# GPU 없는 환경 (팀원 서버 / Mac / CPU) — docker-compose.yml
docker compose up          # 포그라운드 (로그 보임)
docker compose up -d       # 백그라운드
docker compose down        # 종료

# GPU 서버 (NVIDIA) — docker-compose.gpu.yml 하나만
docker compose -f docker-compose.gpu.yml up
```
→ `http://<서버>:8000/docs` 접속.

- **GPU 여부에 따라 파일 하나만 고르면 됨** (각 파일이 독립 실행 가능):
  - GPU 없음 → `docker-compose.yml` → `docker compose up`
  - GPU 있음(NVIDIA) → `docker-compose.gpu.yml` → `docker compose -f docker-compose.gpu.yml up`
  - 팀원은 자기 환경에 맞는 **파일 하나만 받으면** 됩니다.
- **누끼 저장 위치**: 호스트의 `./static` 폴더(= 컨테이너 `/app/static`)에 쌓임. 컨테이너를 지워도 유지되고, **같은 폴더를 백엔드가 공유**해 `/analyze_path`가 준 경로로 직접 읽을 수 있음.
  - `./static` 폴더는 `docker compose up` 시 **자동 생성**됨.
  - 저장 위치를 고정하려면 `docker-compose.yml`의 `volumes`를 절대경로로: `- /srv/static:/app/static` (백엔드가 읽을 경로와 맞출 것).
- **백엔드와 볼륨 공유**: `docker-compose.yml`의 주석 처리된 `backend:` 서비스에 같은 `- ./static:/app/static`를 걸면 됨.
- **로컬 빌드**: `docker-compose.yml`에서 `image:` 대신 `build: .` 사용 → `docker compose up --build`.

### B. docker run (단발성)

```bash
# NVIDIA GPU 서버 (x86_64 + nvidia-container-toolkit)
docker pull rabbitjun/clothing-api
docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/static:/app/static" rabbitjun/clothing-api

# GPU 없는 환경 (CPU 폴백, 느림)
docker run --rm -p 8000:8000 -v "$(pwd)/static:/app/static" rabbitjun/clothing-api
```

> 직접 빌드: `docker build -t clothing-api . && docker run --rm --gpus all -p 8000:8000 clothing-api`
> HF 모델 캐시 재사용(다운로드 생략): `-v ~/.cache/huggingface:/root/.cache/huggingface`

### C. 로컬 (개발용)

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
main.py            FastAPI 앱 (/analyze, /analyze_path, /segment, /segment2, /attrs, /color)
seg1.py            segformer 누끼          detect.py         yolos 검출
sam_cutout.py      yolos + SAM2           ground_detect.py  grounding-dino 검출
ground_sam.py      grounding-dino + SAM2  seg_rembg.py      rembg
pattern.py         FashionCLIP 속성        colors.py         K-means 대표색
device_check.py    디바이스 선택(cuda/mps/cpu)

bench.py           속도·VRAM 벤치마크
generate.py        모델별 부위 누끼 이미지 생성 (품질 비교용)
extract_info.py    이미지별 속성+색상 추출 → output/info.json
decode.py          /segment2 base64 응답 → PNG 디코드 헬퍼

Dockerfile · docker-compose.yml · docker-compose.gpu.yml · requirements.txt · .dockerignore
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
