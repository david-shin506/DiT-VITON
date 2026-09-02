# TinyDiT VITON

`viton-fromscratch (3).ipynb`의 모델, 두 데이터셋 전처리, VAE latent 캐시 생성,
학습, W&B 로깅, 체크포인트 저장/재개 및 추론 기능을 실행 가능한 Python 모듈로
분리한 프로젝트입니다. 원본 노트북은 비교와 재현을 위해 수정하지 않습니다.

## 구조

```text
config/
  dataset.yaml       # 데이터셋 절대경로, 마스크/크롭 설정
  path.yaml          # latent 캐시, 체크포인트, 추론 입출력 경로
  wandb.yaml         # W&B 프로젝트와 실행 설정(API 키 제외)
  model.yaml         # VAE와 DiT 구조
  train.yaml         # 학습 하이퍼파라미터
src/
  model/
    attention.py     # SelfAttention, CrossAttention
    embeddings.py    # timestep 및 2D sin/cos 임베딩
    blocks.py        # FFN, TransformerBlock, FinalLayer
    dit.py           # DiT
    ema.py           # EMA
    flow_matching.py # FMLoss와 Euler sampling
  preprocessing/
    fashionpedia_preprocess.py
    people_clothing_segmentation_preprocess.py
    common.py        # fallback mask와 multi-GPU VAE encoder
  vae_preprocessing/
    latent_save.sh   # 데이터셋 이름으로 cache builder 실행
  data.py            # latent cache 학습 데이터셋
  train.py           # 학습, 샘플 로깅, 체크포인트 저장
  inference.py       # VITON-HD 추론
```

## 환경 준비

Kaggle/Linux, Python 3.10 이상, CUDA 지원 GPU를 기준으로 합니다.

```bash
cd /path/to/TinyDiT_VITON
python -m pip install -r requirements.txt
```

Hugging Face에서 `madebyollin/sdxl-vae-fp16-fix`를 처음 읽을 때는 인터넷 또는
이미 다운로드된 모델 캐시가 필요합니다.

## 설정

### 데이터셋

`config/dataset.yaml`에는 현재 노트북에서 사용한 절대경로가 들어 있습니다.

- Fashionpedia: `/kaggle/input/datasets/pchhalotre321chh/fashionpedia-dataset`
- People Clothing Segmentation: `/kaggle/input/datasets/rajkumarl/people-clothing-segmentation`

Kaggle Dataset slug나 마운트 위치가 바뀌면 이 두 `root`와 PCS의 `labels_csv`를
수정합니다. 이미지 크기, latent 크기, VAE scale과 각 데이터셋의 마스크 설정도
같은 파일에서 관리합니다.

### 출력 경로

`config/path.yaml`의 `paths.cache`가 `build_cache` 결과의 저장 위치입니다.
노트북과 같은 초기값은 다음과 같습니다.

```yaml
cache:
  fashionpedia: /kaggle/working/latents.ckpt
  people_clothing_segmentation: /kaggle/working/latents_pcs.ckpt
```

체크포인트 디렉터리, 재개할 체크포인트, 추론 체크포인트 및 추론 결과 디렉터리도
`path.yaml`에서 변경할 수 있습니다.

### W&B

API 키는 파일에 저장하지 않습니다. Kaggle Secret 또는 환경변수로 설정합니다.

```bash
export WANDB_API_KEY="your-api-key"
```

프로젝트명, run 이름, entity, group, tags, mode, resume 정책 등은
`config/wandb.yaml`에서 설정합니다. W&B를 사용하지 않을 때는 `enabled: false`로
바꿉니다.

## latent 캐시 생성

프로젝트 루트에서 실행합니다. 기존 캐시가 있으면 자동으로 건너뜁니다.

```bash
bash src/vae_preprocessing/latent_save.sh --dataset-name fashionpedia
bash src/vae_preprocessing/latent_save.sh --dataset-name people_clothing_segmentation
```

PCS는 `pcs` 별칭도 지원합니다.

```bash
bash src/vae_preprocessing/latent_save.sh --dataset-name pcs
```

노트북과 같은 기본 실행 인자는 `batch=24`, `chunk=32`, `workers=4`, `seed=0`입니다.
처리 진행률은 기본적으로 1000개마다 `현재 처리 수/전체 수` 형식으로 출력되며,
`config/dataset.yaml`의 `common.progress_every`에서 간격을 변경할 수 있습니다.
필요하면 Python cache builder로 그대로 전달할 수 있습니다.

```bash
bash src/vae_preprocessing/latent_save.sh \
  --dataset-name fashionpedia \
  --batch 24 --chunk 32 --workers 4 --seed 0 \
  --progress-every 500 --force
```

`--progress-every`는 해당 실행에서 YAML의 진행률 간격을 덮어씁니다. 마지막 처리
수는 설정 간격과 무관하게 항상 출력됩니다. `--force`는 기존 캐시를 다시 생성합니다.
일회성으로 저장 경로를 덮어쓸 때는 `--output /absolute/path/cache.ckpt`를 사용할 수
있습니다.

## 학습

두 캐시가 모두 존재하면 이어 붙여 학습합니다. PCS 캐시가 없으면 노트북과 같이
Fashionpedia 캐시만 사용합니다.

```bash
python -m src.train
```

학습 설정은 `config/train.yaml`, 모델 구조는 `config/model.yaml`에서 변경합니다.
`paths.resume_checkpoint`가 존재하면 `model`과 `ema` 가중치를 불러옵니다. 명령행에서
일시적으로 덮어쓸 수도 있습니다.

```bash
python -m src.train --resume /absolute/path/latest.ckpt
```

노트북과 동일하게 체크포인트는 다음 키를 갖습니다.

```text
model: model.state_dict()
ema:   ema.model.state_dict()
```

기본 저장 파일은 `/kaggle/working/ckpt/latest.ckpt`입니다. W&B에는 loss, learning
rate, gradient norm, 처리량, GPU 메모리와 fixed/random sample grid가 기록됩니다.

## 추론

인자를 생략하면 `config/path.yaml`의 기본 체크포인트와 VITON-HD 샘플
`agnostic-v3.2/00036_00.jpg`를 사용합니다.

```bash
python -m src.inference
```

명시적으로 입력과 결과를 지정하는 예시는 다음과 같습니다.

```bash
python -m src.inference \
  --agnostic /absolute/path/agnostic-v3.2/00036_00.jpg \
  --checkpoint /absolute/path/latest.ckpt \
  --output /kaggle/working/inference/result.png \
  --steps 50 --seed 0
```

기본 모델은 노트북 학습과 동일하게 `cross_attention_end: -1`이므로 agnostic
inpainting을 수행합니다. 별도 garment latent를 사용하는 모델을 학습한 경우
`model.yaml`의 `cross_attention_end`를 0 이상으로 맞추고 `--cloth`를 전달합니다.

## 원본 노트북 대응

- 모델 셀 → `src/model/`
- Fashionpedia YOLO bbox 마스크 및 `build_cache` →
  `src/preprocessing/fashionpedia_preprocess.py`
- People Clothing Segmentation crop/segmentation 마스크, prescan 및
  `build_pcs_cache` → `src/preprocessing/people_clothing_segmentation_preprocess.py`
- `InpaintDS`, 두 캐시 결합 → `src/data.py`
- `FMLoss`, EMA, 학습 루프, W&B sample grid, 체크포인트 → `src/model/` 및
  `src/train.py`
- VITON-HD gray-mask 복원과 추론 → `src/inference.py`

캐시 key(`z_full`, `z_agn`, `masks`, `n_valid`, `scale`)와 모델 parameter 이름을
유지하므로 기존 노트북에서 생성한 캐시와 체크포인트를 그대로 사용할 수 있습니다.
