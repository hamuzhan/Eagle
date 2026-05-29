# AGENTS.md

Eagle is NVIDIA's open-source family of vision-language model projects, including the original Eagle VLMs, Eagle 2.5 long-context image/video VLMs, and LocateAnything grounding models.

This is a Python/PyTorch research codebase with training, inference, evaluation, Streamlit/Gradio demos, and TensorRT-LLM deployment utilities. Most meaningful runs require NVIDIA GPUs, Hugging Face/model access, model checkpoints, and prepared external datasets; ask before launching training, full evaluation, Docker/TensorRT builds, MagiAttention builds, or large downloads.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it. It contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CONTRIBUTING.md` before making code changes.
- Follow the style of the file, module, and subproject you are editing; this repo contains copied/adapted code from LLaVA, Hugging Face Transformers, Qwen, InternVL/OpenGVLab, LMMs-Eval, Microsoft, Unsloth, and others.
- Preserve existing copyright and license headers. For new Python files, use the NVIDIA header style used by neighboring files in the same subproject, and do not remove third-party attribution blocks.
- Use `git commit -s` for commits. DCO sign-off is required; rely on `git` to add the sign-off and do not manually add AI attribution or co-authors unless explicitly requested.
- Run formatting before submitting code when dependencies are installed: `black .`, `isort .`, and `flake8 .` from the relevant subproject.
- Do not check in model weights, datasets, generated checkpoints, `work_dirs*`, `local_playground*`, `playground*`, `pretrained*`, `wandb*`, logs, `.prepare.json`, parquet/LMDB outputs, `*.onnx`, `*.pth`, TensorRT engines, or Hugging Face cache contents.
- `Eagle2_5/` and `Embodied/` both define a top-level Python package named `eaglevl`. Do not install both editable in the same environment unless you intentionally want one to shadow the other.
- Most training/evaluation commands assume CUDA GPUs, correct PyTorch/CUDA wheels, Hugging Face access, model checkpoints, and prepared datasets. Verify paths and environment variables before running.

## Project Boundaries

| Path | Package / role | Setup notes |
|------|----------------|-------------|
| `Eagle/` | package `eagle`; original Eagle/Eagle 2, LLaVA-style training/inference/eval | `pip install -r requirements.txt && pip install .`; pins include `torch==2.3.1`, `transformers==4.44.2`, `tokenizers==0.19.1` |
| `Eagle2_5/` | project `eagle_vl`, imports `eaglevl`; long-context image/video VLM | docs recommend Python 3.10, `torch==2.5.0` CUDA wheel, `flash-attn==2.4.2`, then `pip install -e .`; pins include `transformers==4.51.0`, `tokenizers==0.21.1`, `deepspeed==0.16.5`, `triton==3.1.0` |
| `Embodied/` | project `locate_anything`, imports `eaglevl`; LocateAnything grounding/detection | `pip install -e .`; pins include `transformers==4.57.1`, `tokenizers==0.22.0`, `deepspeed==0.15.4`, older `gradio/httpx` |

## Common Commands

Run commands from the directory shown in `Workdir`; avoid assuming the repo root is the working directory.

| Task | Workdir | Command |
|------|---------|---------|
| Format Python | relevant subproject | `black . && isort . && flake8 .` |
| Docs-only whitespace check | repo root | `git diff --check` |
| Original Eagle install | `Eagle/` | `pip install -r requirements.txt && pip install .` |
| Original Eagle inference | `Eagle/` | `python predict_demo.py` |
| Original Eagle Gradio demo | `Eagle/` | `python gradio_demo.py --model-path ${MODEL_CKPT} --conv-mode vicuna_v1` |
| Original Eagle LMMs-Eval suite | `Eagle/` | `bash scripts/eval_lmms_eval/eval-mme-seed-pope-sqa-gqa-ocrbench-textvqa-chartqa.sh $REPO_ID_OR_LOCAL_PATH $MODEL_NAME $CONV_MODE` |
| Eagle 2.5 install | `Eagle2_5/` | `pip install -e .` |
| Eagle 2.5 prepare data | `Eagle2_5/` | `bash shell/prepare.sh <recipe.json>`; pass an explicit recipe because the script default is `local_playground/recipe/stage1.json` |
| Eagle 2.5 finetune | `Eagle2_5/` | `GPUS=8 bash shell/train_stage2.sh 1 <output_dir>`; inspect and replace internal default checkpoint/meta paths first |
| Eagle 2.5 Streamlit demo | `Eagle2_5/streamlit_demo/` | `bash start_demo.sh` |
| LocateAnything install | `Embodied/` | `pip install -e .` |
| LocateAnything finetune | `Embodied/` | `torchrun --nproc_per_node=8 eaglevl/train/locany_finetune_magi_stream.py --model_name_or_path nvidia/LocateAnything-3B --meta_path ./locany_recipe/my_recipe.json --output_dir work_dirs/my_sft --attn_implementation magi --deepspeed deepspeed_configs/zero_stage2_config.json` |
| LocateAnything COCO eval | `Embodied/` | `bash evaluation/scripts/eval_coco.sh --model_path <model> --test_jsonl <EvalData/_annotations/box_eval/COCO.jsonl> --image_root <EvalData> --coco_json <EvalData/coco/instances_val2017.json> --output_dir <out>` |
| LocateAnything grounding eval | `Embodied/` | `bash evaluation/scripts/eval_grounding.sh --dataset Dense200 --eval_type box_eval --model_path <model> --image_root <EvalData> --output_base <out_base>` |

### Installation Notes

| Subproject | Recommended Setup | Notes |
|------------|-------------------|-------|
| `Eagle/` | Python 3.10 conda env, `pip install -r requirements.txt`, `pip install .` | Training scripts use DeepSpeed configs under `Eagle/scripts/`; additional `flash-attn` is recommended for training. |
| `Eagle2_5/` | Python 3.10, PyTorch matching CUDA, `flash-attn==2.4.2`, `pip install -e .` | See `Eagle2_5/document/1.installing.md`; dependencies are pinned in `pyproject.toml`. |
| `Embodied/` | Python 3.10, `pip install -e .` | LocateAnything dependencies are pinned in `Embodied/pyproject.toml`; MagiAttention is optional but required for long context on supported GPUs. |

### GPU and Attention Notes

- Eagle 2.5 long-context training relies on distributed sequence/context parallelism and DeepSpeed.
- LocateAnything supports `--attn_implementation magi` for Hopper/Blackwell GPUs and `--attn_implementation sdpa` for other GPUs.
- MagiAttention only supports Hopper/Blackwell and is required for LocateAnything long-context training around 16K-32K+ tokens.
- On A100, L40, or other non-Hopper/Blackwell GPUs, use `sdpa` and keep LocateAnything sequence lengths around 4K.
- `flash-attn` installation must match the local CUDA/PyTorch stack; do not upgrade PyTorch, Transformers, Tokenizers, DeepSpeed, Triton, Gradio, or HTTPX casually because versions differ by subproject and are tightly coupled.

## Repository Layout

| Path | Role |
|------|------|
| `README.md` | Top-level project overview and model family landing page. |
| `CONTRIBUTING.md` | Contribution rules, formatting expectations, PR workflow, and DCO sign-off requirement. |
| `Eagle/` | Original Eagle/Eagle 2 codebase built on LLaVA-style multimodal LLM training, inference, demos, and LMMs-Eval. |
| `Eagle2_5/` | Eagle 2.5 long-context image/video VLM codebase with data preparation, training, Streamlit demo, and TensorRT-LLM deployment. |
| `Embodied/` | LocateAnything visual grounding and detection codebase with PBD/MTP inference, training, and evaluation. |

## Architecture

### Original Eagle (`Eagle/`)

Eagle is a vision-centric multimodal LLM family with mixture-of-encoders. It is derived from LLaVA-style training and generation, with multiple vision towers fused through a multimodal projector.

| Component | Key Path |
|-----------|----------|
| Model architecture | `Eagle/eagle/model/eagle_arch.py` |
| LLM wrapper | `Eagle/eagle/model/language_model/eagle_llama.py` |
| Model loading | `Eagle/eagle/model/builder.py` |
| Training entry point | `Eagle/train.py` and `Eagle/train_mem.py` |
| Trainer customization | `Eagle/eagle/train/eagle_trainer.py` |
| Vision encoders | `Eagle/eagle/model/multimodal_encoder/` |
| Projector builder | `Eagle/eagle/model/multimodal_projector/` |
| Conversation templates | `Eagle/eagle/conversation.py` |
| Inference example | `Eagle/predict_demo.py` |
| Evaluation wrapper | `Eagle/evaluate_lmms_eval.py` and `Eagle/lmms_eval/` |

Request flow:

```text
Image + prompt -> conversation template -> tokenizer_image_token/process_images
    -> EagleLlamaForCausalLM.generate -> decoded answer
```

Training flow:

```text
JSON annotations + image folder -> train.py/train_mem.py -> EagleTrainer
    -> EagleMetaForCausalLM -> vision tower(s) + projector + language model
```

### Eagle 2.5 (`Eagle2_5/`)

Eagle 2.5 is a long-context VLM for high-resolution images, multi-page documents, and long videos. It uses information-first sampling, online packing, context/sequence parallel utilities, fused ops, and DeepSpeed.

| Component | Key Path |
|-----------|----------|
| Config | `Eagle2_5/eaglevl/model/eagle2_5/configuration_eagle2_5_vl.py` |
| Model | `Eagle2_5/eaglevl/model/eagle2_5/modeling_eagle2_5_vl.py` |
| Training entry point | `Eagle2_5/eaglevl/train/eagle_2_5_vl_finetune.py` |
| Data preparation | `Eagle2_5/eaglevl/train/data_prepare.py` and `Eagle2_5/shell/prepare.sh` |
| Dataset and packing | `Eagle2_5/eaglevl/train/dataset.py` |
| Training args | `Eagle2_5/eaglevl/train/arguments.py` |
| Sequence/context parallel | `Eagle2_5/eaglevl/sp_utils/` |
| Fused/monkey patches | `Eagle2_5/eaglevl/patch/` |
| Launch scripts | `Eagle2_5/shell/` |
| DeepSpeed configs | `Eagle2_5/deepspeed_configs/` |
| Streamlit demo | `Eagle2_5/streamlit_demo/` |
| TensorRT-LLM deployment | `Eagle2_5/deployment/` |

Data flow:

```text
recipe JSON + JSONL/media -> shell/prepare.sh -> .prepare.json + parquet metadata
    -> shell/train_stage*.sh -> eaglevl/train/eagle_2_5_vl_finetune.py
```

Inference flow:

```text
AutoProcessor + AutoModel.from_pretrained("nvidia/Eagle-2.5-8B", trust_remote_code=True)
    -> process_vision_info(images/videos) -> model.generate -> decoded answer
```

Script-specific gotchas:

- `shell/prepare.sh` defaults to `local_playground/recipe/stage1.json`, tokenizer `Qwen/Qwen3-1.7B`, max length `49152`, FPS `4`, min frames `8`, and max frames `24`; pass an explicit recipe and review frame/token settings before preparing data.
- `shell/train_stage1.sh` and `shell/train_stage2.sh` include internal default paths such as `local_playground/recipe/stage1.prepared.json`, `pretrained/siglip2-so400m-patch14-448`, and `work_dirs/eagle_er-qwen3_8B-Siglip2_400M_stage1_5_128gpu_v19`; replace these before running.
- Docs may show `playground/...` examples while scripts currently use `local_playground/...`; trust executable scripts over prose when they conflict.

### LocateAnything (`Embodied/`)

LocateAnything is a visual grounding/detection model based on Eagle, MoonViT, Qwen2.5/Qwen3, and Parallel Box Decoding. It supports fast MTP mode, slow NTP mode, and hybrid fallback.

| Component | Key Path |
|-----------|----------|
| Worker API | `Embodied/locateanything_worker.py` |
| Config | `Embodied/eaglevl/model/locany/configuration_locateanything.py` |
| Model | `Embodied/eaglevl/model/locany/modeling_locateanything.py` |
| HF utility copy | `Embodied/eaglevl/utils/locany/` |
| Training entry point | `Embodied/eaglevl/train/locany_finetune_magi_stream.py` |
| Training args | `Embodied/eaglevl/train/arguments.py` |
| Dataset and streaming packing | `Embodied/eaglevl/train/dataset.py` |
| Sequence parallel utilities | `Embodied/eaglevl/sp_utils/` |
| Evaluation scripts | `Embodied/evaluation/` |
| DeepSpeed configs | `Embodied/deepspeed_configs/` |

Inference flow:

```text
PIL image + grounding prompt -> LocateAnythingWorker.predict
    -> AutoProcessor.process_vision_info -> LocateAnythingForConditionalGeneration.generate
    -> box/point token output
```

Output conventions:

| Output | Format |
|--------|--------|
| Box | `<ref>label</ref><box><x1><y1><x2><y2></box>` |
| Point | `<box><x><y></box>` |
| No object | `<box>none</box>` |

Coordinates are normalized integer tokens in `[0, 1000]`; divide by 1000 and multiply by image width/height for pixel coordinates.

LocateAnything data/eval gotchas:

- Recipes point `--meta_path` at JSON mapping dataset names to `annotation`, `root`, `repeat_time`, and optional `data_augment`; JSONL media paths are resolved relative to `root`.
- Use `<image-1>`, `<image-2>`, etc. in JSONL conversations; if omitted, image placeholders may be prepended automatically.
- Detection prompts use `</c>` as the category separator.
- `generation_mode` options are `fast`, `slow`, and `hybrid`; `hybrid` is the documented/default stability path.
- `Embodied/evaluation/README.md` requires downloading/installing Rex-Omni `fastevaluate` plus `shapely` before COCO/LVIS metrics.

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| Hugging Face model integration | Custom configs inherit `PretrainedConfig`; models inherit `PreTrainedModel` and often `GenerationMixin`; `trust_remote_code=True` is expected for released Eagle 2.5 and LocateAnything checkpoints. |
| LLaVA-style multimodal prompting | Original Eagle uses conversation templates plus image tokens; do not change template behavior without checking downstream training/eval compatibility. |
| Two-stage training | Original Eagle uses pretraining for the multimodal projector followed by supervised fine-tuning. |
| Data recipes | Eagle 2.5 and LocateAnything use recipe JSON files pointing to JSONL annotations and media roots; recipe shape differs between subprojects. |
| Online/streaming packing | Eagle 2.5 and LocateAnything reduce padding waste via packing buffers and token budgets; changes can affect reproducibility and checkpoint resume. |
| Distributed training | Launch scripts use `torchrun`, DeepSpeed ZeRO configs, NCCL settings, and sometimes SLURM environment variables. |
| Fused kernels and monkey patches | Fused ops and monkey patches are performance-sensitive; keep diffs minimal and verify imports/training smoke tests when changed. |
| Deployment split | Eagle 2.5 TensorRT deployment builds vision ONNX/TensorRT engines separately from TensorRT-LLM LLM engines. |

## Key Files

| File | Role |
|------|------|
| `CONTRIBUTING.md` | Contribution, formatting, commit, PR, and DCO rules. |
| `Eagle/README.md` | Original Eagle install, training, inference, evaluation, and demo docs. |
| `Eagle/requirements.txt` | Original Eagle dependency pins. |
| `Eagle/setup.py` | Original Eagle package metadata. |
| `Eagle/eagle/model/eagle_arch.py` | Original Eagle multimodal architecture core. |
| `Eagle/eagle/model/language_model/eagle_llama.py` | Eagle Llama config/model registration. |
| `Eagle/eagle/model/builder.py` | Original Eagle checkpoint loading and vision tower setup. |
| `Eagle/train.py` | Main original Eagle training logic and argument dataclasses. |
| `Eagle/train_mem.py` | Memory-optimized original Eagle training entry point used by scripts. |
| `Eagle/predict_demo.py` | Minimal original Eagle inference example. |
| `Eagle/gradio_demo.py` | Original Eagle Gradio demo. |
| `Eagle/evaluate_lmms_eval.py` | Local LMMs-Eval entry point. |
| `Eagle/scripts/` | SLURM-oriented training/eval scripts; expect dataset/checkpoint env vars. |
| `Eagle2_5/README.md` | Eagle 2.5 overview and links to install/data/train/demo/inference docs. |
| `Eagle2_5/pyproject.toml` | Eagle 2.5 dependency pins and packaging. |
| `Eagle2_5/document/0.onboarding.md` | Recommended Eagle 2.5 workflow. |
| `Eagle2_5/document/1.installing.md` | Eagle 2.5 installation details. |
| `Eagle2_5/document/2.preparing_playground.md` | Eagle 2.5 data format and preparation docs. |
| `Eagle2_5/document/3.training.md` | Eagle 2.5 training launch docs. |
| `Eagle2_5/document/5.inference.md` | Eagle 2.5 inference examples. |
| `Eagle2_5/eaglevl/model/eagle2_5/modeling_eagle2_5_vl.py` | Eagle 2.5 model implementation. |
| `Eagle2_5/eaglevl/train/data_prepare.py` | Eagle 2.5 recipe normalization, filtering, token length, video frame, and parquet metadata prep. |
| `Eagle2_5/eaglevl/train/eagle_2_5_vl_finetune.py` | Eagle 2.5 finetuning entry point. |
| `Eagle2_5/shell/prepare.sh` | Eagle 2.5 data preparation launcher. |
| `Eagle2_5/shell/train_stage*.sh` | Eagle 2.5 training launchers with internal default paths. |
| `Eagle2_5/deployment/README.md` | TensorRT/TensorRT-LLM deployment guide. |
| `Embodied/README.md` | LocateAnything overview, quick start, training, and evaluation docs. |
| `Embodied/pyproject.toml` | LocateAnything dependency pins and packaging. |
| `Embodied/locateanything_worker.py` | User-facing LocateAnything inference worker. |
| `Embodied/document/TRAINING.md` | LocateAnything full training guide. |
| `Embodied/document/DATA_PREPARATION.md` | LocateAnything JSONL/recipe/token format guide. |
| `Embodied/document/STREAMING_PACKING.md` | LocateAnything streaming packing details. |
| `Embodied/evaluation/README.md` | LocateAnything evaluation setup and commands. |
| `Embodied/evaluation/scripts/` | LocateAnything COCO/LVIS/grounding/ScreenSpot-Pro evaluation launchers. |
| `Embodied/eaglevl/train/locany_finetune_magi_stream.py` | LocateAnything training entry point. |
| `Embodied/eaglevl/model/locany/modeling_locateanything.py` | LocateAnything model implementation. |

## Anti-Patterns / Gotchas

- Do not assume a single project root for all commands. `Eagle/`, `Eagle2_5/`, and `Embodied/` each have their own install flow, dependency pins, scripts, and expected working directory.
- Do not install `Eagle2_5` and `Embodied` into the same editable environment unless you have handled the `eaglevl` package collision.
- Do not silently upgrade pinned dependencies like `transformers`, `tokenizers`, `torch`, `deepspeed`, `triton`, `gradio`, or `httpx`; model code and remote-code checkpoints may depend on exact versions.
- Do not alter prompt templates, image/video placeholders, special tokens, coordinate token formats, or LocateAnything output formats without updating training data, inference parsing, and evaluation code.
- Do not replace vendor-derived files with upstream versions unless the task explicitly requires it; this repo includes local modifications and attribution blocks.
- Do not launch training scripts with placeholder/internal paths. Replace `PATH_TO_*`, `playground/...`, `local_playground/...`, `locany_recipe/...`, and `path/to/...` values first.
- Do not run LocateAnything `magi` attention on unsupported GPUs. Use `sdpa` on non-Hopper/Blackwell hardware.
- Do not assume tests are lightweight. Most validation here is model-, GPU-, and dataset-dependent.
- Do not commit generated `.prepare.json`, parquet/LMDB data, extracted evaluation datasets, checkpoints, `training_log.txt`, TensorRT engines, or W&B output unless explicitly requested.
- Do not change public worker APIs such as `LocateAnythingWorker` methods casually; update README examples and downstream callers if needed.
- Do not remove `trust_remote_code=True` from Eagle 2.5 or LocateAnything examples unless you verified released checkpoints still load.

## Development Workflow

1. Identify which subproject owns the change: `Eagle/`, `Eagle2_5/`, or `Embodied/`.
2. Read the relevant README and docs before editing.
3. Install only the relevant subproject in an isolated environment.
4. Make the smallest correct change, preserving local style and attribution headers.
5. Run targeted formatting and the lightest meaningful validation available.
6. For training/eval changes, document the exact GPU, command, dataset/model paths, and whether the run was skipped due to missing hardware or data.
7. Commit with `git commit -s` only when explicitly requested.

## Testing / Validation

There is no single universal unit-test suite in this repo. Prefer targeted validation based on the files touched.

| Change Area | Suggested Validation |
|-------------|----------------------|
| Docs only | Review rendered Markdown if relevant and run `git diff --check`. |
| Formatting-only Python changes | `black . && isort . && flake8 .` in the relevant subproject. |
| Original Eagle inference/model loading | `python predict_demo.py` in `Eagle/` when GPU and checkpoint access are available. |
| Original Eagle eval code | Use the intended script under `Eagle/scripts/eval_lmms_eval/` with the required datasets and model path. |
| Eagle 2.5 data prep | `bash shell/prepare.sh <small_recipe>` in `Eagle2_5/` when GPU/data paths are available. |
| Eagle 2.5 training code | `GPUS=<n> bash shell/train_stage2.sh <nodes> <output_dir>` in `Eagle2_5/` on a small/debug run after confirming paths and hardware. |
| Eagle 2.5 inference | Follow `Eagle2_5/document/5.inference.md` with `nvidia/Eagle-2.5-8B` or a local checkpoint. |
| Eagle 2.5 deployment | Follow `Eagle2_5/deployment/README.md`; requires TensorRT 10.11 Docker, TensorRT-LLM commit `7c828d767f12dd505187e626357a80da9f05a99a`, and `deployment/tensorrt_llm.patch`. |
| LocateAnything worker | Run a small `LocateAnythingWorker` inference on one image when GPU/model access is available. |
| LocateAnything training | Run a short `torchrun` smoke test with a tiny recipe and the appropriate attention backend. |
| LocateAnything evaluation | Use scripts under `Embodied/evaluation/scripts/` with prepared EvalData; COCO/LVIS also require `fastevaluate`. |

## Branching and PRs

- Main upstream repository: `https://github.com/NVlabs/Eagle`.
- In this workspace, `origin` may point to the user's fork; push only when explicitly requested.
- Start feature work from an issue when contributing upstream, per `CONTRIBUTING.md`.
- Keep PRs focused on one concern. Split unrelated model, data, eval, deployment, and documentation changes.
- PRs marked work-in-progress should use `[WIP]` in the title and remove it when ready.
- Before any `gh` command, confirm the intended GitHub host, target repository, and authentication config. If a custom `GH_CONFIG_DIR` is specified in local notes or environment, use it.

## Key Documentation

| Topic | Path |
|-------|------|
| Contribution rules | `CONTRIBUTING.md` |
| Top-level model family | `README.md` |
| Original Eagle guide | `Eagle/README.md` |
| Eagle 2.5 onboarding | `Eagle2_5/document/0.onboarding.md` |
| Eagle 2.5 install | `Eagle2_5/document/1.installing.md` |
| Eagle 2.5 data preparation | `Eagle2_5/document/2.preparing_playground.md` |
| Eagle 2.5 training | `Eagle2_5/document/3.training.md` |
| Eagle 2.5 Streamlit demo | `Eagle2_5/document/4.streamlit_demo.md` |
| Eagle 2.5 inference | `Eagle2_5/document/5.inference.md` |
| Eagle 2.5 script arguments | `Eagle2_5/document/explain_script_arguments.md` |
| Eagle 2.5 LMDB reading | `Eagle2_5/document/how_to_use_lmdb_to_read_images.md` |
| Eagle 2.5 deployment | `Eagle2_5/deployment/README.md` |
| LocateAnything guide | `Embodied/README.md` |
| LocateAnything training | `Embodied/document/TRAINING.md` |
| LocateAnything data preparation | `Embodied/document/DATA_PREPARATION.md` |
| LocateAnything streaming packing | `Embodied/document/STREAMING_PACKING.md` |
| LocateAnything results | `Embodied/document/RESULTS.md` |
| LocateAnything evaluation | `Embodied/evaluation/README.md` |
