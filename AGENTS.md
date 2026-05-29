# AGENTS.md

Eagle is a three-subproject Python/PyTorch research repo for NVIDIA VLMs. Most non-doc runs need CUDA GPUs, Hugging Face/model access, checkpoints, and prepared external datasets; ask before launching training, full evaluation, Docker/TensorRT builds, MagiAttention builds, or large downloads.

## Required Workflow

- Read `CONTRIBUTING.md` before code changes; preserve existing copyright, license, and third-party attribution headers.
- Follow the edited subproject's local style; code is adapted from LLaVA, Hugging Face Transformers, Qwen, InternVL, LMMs-Eval, Microsoft, Unsloth, and others.
- Use `git commit -s` if committing; DCO sign-off is required.
- No CI workflows, pre-commit config, or repo-wide pytest config are checked in. Do not invent a universal test command.
- Formatting expectation from `CONTRIBUTING.md`: run `black .`, `isort .`, and `flake8 .` from the relevant subproject when dependencies are installed. For docs-only changes, `git diff --check` is the lightest useful check.
- Do not commit weights, datasets, generated checkpoints, `work_dirs*`, `local_playground*`, `playground*`, `pretrained*`, `wandb*`, logs, `.prepare.json`, parquet/LMDB outputs, `*.onnx`, `*.pth`, TensorRT engines, or Hugging Face caches.

## Project Boundaries

| Path | Package / role | Setup notes |
| --- | --- | --- |
| `Eagle/` | package `eagle`; original Eagle/Eagle 2, LLaVA-style training/inference/eval | `pip install -r requirements.txt && pip install .`; pins include `torch==2.3.1`, `transformers==4.44.2`, `tokenizers==0.19.1` |
| `Eagle2_5/` | project `eagle_vl`, imports `eaglevl`; long-context image/video VLM | docs recommend Python 3.10, `torch==2.5.0` CUDA wheel, `flash-attn==2.4.2`, then `pip install -e .`; pins `transformers==4.51.0`, `deepspeed==0.16.5`, `triton==3.1.0` |
| `Embodied/` | project `locate_anything`, imports `eaglevl`; LocateAnything grounding/detection | `pip install -e .`; pins `transformers==4.57.1`, `tokenizers==0.22.0`, `deepspeed==0.15.4`, older `gradio/httpx` |

`Eagle2_5/` and `Embodied/` both provide a top-level `eaglevl` package. Do not install both editable in the same environment unless you intentionally want one to shadow the other.

## Commands

Run commands from the subproject directory shown by the path.

| Task | Command |
| --- | --- |
| Original Eagle install | `pip install -r requirements.txt && pip install .` in `Eagle/` |
| Original Eagle inference smoke | `python predict_demo.py` in `Eagle/` |
| Original Eagle Gradio | `python gradio_demo.py --model-path ${MODEL_CKPT} --conv-mode vicuna_v1` in `Eagle/` |
| Original Eagle LMMs-Eval | `bash scripts/eval_lmms_eval/eval-mme-seed-pope-sqa-gqa-ocrbench-textvqa-chartqa.sh $REPO_ID_OR_LOCAL_PATH $MODEL_NAME $CONV_MODE` in `Eagle/` |
| Eagle 2.5 install | `pip install -e .` in `Eagle2_5/` |
| Eagle 2.5 data prep | `bash shell/prepare.sh <recipe.json>` in `Eagle2_5/`; pass an explicit recipe, since script defaults point at `local_playground/recipe/stage1.json` |
| Eagle 2.5 finetune | `GPUS=8 bash shell/train_stage2.sh 1 <output_dir>` in `Eagle2_5/`; inspect and replace internal default checkpoint/meta paths first |
| Eagle 2.5 Streamlit | `bash start_demo.sh` in `Eagle2_5/streamlit_demo/` |
| LocateAnything install | `pip install -e .` in `Embodied/` |
| LocateAnything finetune | `torchrun --nproc_per_node=8 eaglevl/train/locany_finetune_magi_stream.py --model_name_or_path nvidia/LocateAnything-3B --meta_path ./locany_recipe/my_recipe.json --output_dir work_dirs/my_sft --attn_implementation magi --deepspeed deepspeed_configs/zero_stage2_config.json` in `Embodied/` |
| LocateAnything COCO eval | `bash evaluation/scripts/eval_coco.sh --model_path <model> --test_jsonl <EvalData/_annotations/box_eval/COCO.jsonl> --image_root <EvalData> --coco_json <EvalData/coco/instances_val2017.json> --output_dir <out>` in `Embodied/` |
| LocateAnything grounding eval | `bash evaluation/scripts/eval_grounding.sh --dataset Dense200 --eval_type box_eval --model_path <model> --image_root <EvalData> --output_base <out_base>` in `Embodied/` |

## Architecture Pointers

- `Eagle/`: `predict_demo.py` uses `eagle/model/builder.py`, `conversation.py`, `mm_utils.py`, and `EagleLlamaForCausalLM.generate`; training entrypoints are `train.py` and `train_mem.py`, with `eagle/train/eagle_trainer.py`.
- `Eagle/` training scripts under `scripts/` are SLURM-oriented and expect `$SLURM_*`, `$PATH_TO_PRETRAINING_DATA`, `$PATH_TO_SFT_DATA`, and `$PATH_TO_PRETRAINED_PROJECTOR`; do not run them unchanged on a local shell.
- `Eagle2_5/`: JSONL/media recipes go through `shell/prepare.sh` -> `eaglevl/train/data_prepare.py` -> `.prepare.json` plus parquet metadata, then `shell/train_stage*.sh` -> `eaglevl/train/eagle_2_5_vl_finetune.py`.
- `Eagle2_5/shell/*.sh` currently use `local_playground/...` and internal checkpoint defaults even where docs show `playground/...`; trust scripts over prose and replace paths before running.
- `Embodied/`: public inference API is `locateanything_worker.py`; training is `eaglevl/train/locany_finetune_magi_stream.py`; evaluation is under `evaluation/scripts/` and requires prepared EvalData.
- `Embodied/evaluation/README.md` requires downloading/installing Rex-Omni `fastevaluate` plus `shapely` before COCO/LVIS metrics.

## GPU, Data, And Token Gotchas

- LocateAnything `--attn_implementation magi` only supports Hopper/Blackwell GPUs and is needed for 16K-32K+ long context. On A100/L40/other GPUs, use `sdpa` and keep sequence length around 4K.
- `flash-attn`, PyTorch, Transformers, Tokenizers, DeepSpeed, Triton, Gradio, and HTTPX versions differ across subprojects; do not upgrade pins casually.
- Eagle 2.5 and LocateAnything examples load released checkpoints with `trust_remote_code=True`; do not remove it unless you verify checkpoint loading still works.
- Eagle 2.5 data prep calculates token lengths, samples video frames, filters bad samples, and writes parquet metadata; review `shell/prepare.sh` for tokenizer/frame settings before preparing data.
- LocateAnything recipes point `--meta_path` at JSON mapping dataset names to `annotation`, `root`, `repeat_time`, and optional `data_augment`; JSONL paths are resolved relative to `root`.
- LocateAnything outputs boxes as `<ref>label</ref><box><x1><y1><x2><y2></box>`, points as `<box><x><y></box>`, and no-object as `<box>none</box>`; coordinates are integer tokens normalized to `[0, 1000]`.
- Do not alter conversation templates, image/video placeholders, special tokens, coordinate tokens, or LocateAnything output formats without updating training data, inference parsing, and evaluation.
- LocateAnything generation modes are `fast`, `slow`, and `hybrid`; `hybrid` is the documented/default stability path.

## Validation Targets

- Prefer the smallest validation tied to touched files; most full suites are GPU-, checkpoint-, and dataset-dependent.
- Docs only: inspect rendered Markdown if relevant and run `git diff --check`.
- Original Eagle model/inference changes: `python predict_demo.py` in `Eagle/` only when GPU/checkpoint access is available.
- Eagle 2.5 data changes: run `bash shell/prepare.sh <small_recipe>` in `Eagle2_5/` when GPU/data paths are available.
- Eagle 2.5 training changes: use a debug `GPUS=<n> bash shell/train_stage2.sh <nodes> <output_dir>` only after confirming paths and hardware.
- LocateAnything worker changes: run one `LocateAnythingWorker` inference when GPU/model access is available.
- LocateAnything eval changes: use the relevant `Embodied/evaluation/scripts/*.sh` with prepared EvalData; COCO/LVIS also need `fastevaluate`.
- Eagle 2.5 TensorRT deployment requires TensorRT 10.11 Docker, TensorRT-LLM commit `7c828d767f12dd505187e626357a80da9f05a99a`, and `deployment/tensorrt_llm.patch`; do not start builds without confirmation.
