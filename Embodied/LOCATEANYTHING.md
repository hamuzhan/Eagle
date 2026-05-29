# LocateAnything — Codebase Analysis

A deep, code-level walkthrough of the LocateAnything project under `Embodied/`. It is written
for engineers (and coding agents) who need to understand *how the code is wired* before changing
training, inference, data, or evaluation. Where useful, claims point to `file_path:line_number`.

For task-oriented guides see the existing docs: [`README.md`](README.md),
[`document/TRAINING.md`](document/TRAINING.md), [`document/DATA_PREPARATION.md`](document/DATA_PREPARATION.md),
[`document/STREAMING_PACKING.md`](document/STREAMING_PACKING.md),
[`evaluation/README.md`](evaluation/README.md), and the kernel reference
[`eaglevl/KERNELS.md`](eaglevl/KERNELS.md). This file complements them with the architecture and
control flow.

---

## 1. What LocateAnything Is

LocateAnything is a generative vision-language model for **visual grounding and detection**:
object detection, dense detection, referring/phrase grounding, GUI element grounding, OCR/text
localization, document layout, and point localization. Its headline contribution is **Parallel Box
Decoding (PBD)**: instead of emitting a box as a serial stream of coordinate tokens (next-token
prediction, NTP), it decodes a full box `(x1,y1,x2,y2)` as an **atomic block in one forward step**
via Multi-Token Prediction (MTP). This is ~10x faster than serial coordinate decoding while keeping
intra-box geometric coherence.

- Package name: `locate_anything`, import root `eaglevl` (`pyproject.toml:6`).
- Released checkpoint: `nvidia/LocateAnything-3B` (HF, `trust_remote_code=True`).
- Backbone: **MoonViT** vision encoder + **MLP projector** + **Qwen2.5/Qwen3** LLM decoder.

> ⚠️ Package collision: `Embodied/` and `Eagle2_5/` both expose a top-level package named `eaglevl`.
> Do not `pip install -e .` both into the same environment.

---

## 2. The Two-Copy Rule (read this before editing model code)

The model exists as **two near-duplicate copies**, and which one runs depends on the entry point.
This is the single most important thing to understand before touching model code.

| Copy | Path | Role | Used by |
|------|------|------|---------|
| **Training copy** | `eaglevl/model/locany/` + `eaglevl/model/moon_vit/` | In-repo training/runtime model. Fused Liger loss, packing, sequence-parallel hooks. | The training entry point. |
| **Inference / export copy** | `eaglevl/utils/locany/` | The HuggingFace `trust_remote_code` export. Contains the **PBD generation loop**, processor, and image processor. | Released checkpoints loaded via `AutoModel.from_pretrained(..., trust_remote_code=True)`. |

Key consequences:

- The **PBD generate loop lives only in the inference copy** (`eaglevl/utils/locany/modeling_locateanything.py:304-537` and `eaglevl/utils/locany/generate_utils.py`). The training copy's `generate()` just delegates to `language_model.generate()` (`eaglevl/model/locany/modeling_locateanything.py:327-373`).
- The **processor and image processor live only in the inference copy** (`eaglevl/utils/locany/processing_locateanything.py`, `eaglevl/utils/locany/image_processing_locateanything.py`). The training script imports them from there (`eaglevl/train/locany_finetune_magi_stream.py:48-49`).
- At the end of training, the script **copies `utils/locany/*.py` into the output dir** and writes an `auto_map` into `config.json` so the checkpoint becomes self-contained and remote-code loadable (`eaglevl/train/locany_finetune_magi_stream.py:1534-1561`). This is the bridge from training copy → inference copy.
- `KERNELS.md` rule: mirror attention/mask/model changes between `model/locany/` and `utils/locany/` unless a change is intentionally training-only or export-only.

Differences worth knowing:

| Aspect | `model/locany` (training) | `utils/locany` (inference) |
|--------|---------------------------|----------------------------|
| Loss | `LigerFusedLinearCrossEntropyLoss`, only valid labels (`modeling_locateanything.py:296-297`) | plain `CrossEntropyLoss` (`modeling_locateanything.py:255-266`) |
| Packing | `sub_sample_lengths` threaded into LLM (`:245-269`) | none |
| `generate()` | delegates to HF generate | full MTP/NTP/hybrid PBD loop |
| magi attn opt-in | via training script monkey patch | `_autoset_attn_implementation` / `_check_and_adjust_attn_implementation` overrides (`modeling_locateanything.py:68-77`) |

---

## 3. Repository Layout

```
Embodied/
├── README.md                  # overview, quick start, output format
├── pyproject.toml             # deps pinned: transformers 4.57.1, tokenizers 0.22.0, deepspeed 0.15.4
├── locateanything_worker.py   # user-facing inference worker (LocateAnythingWorker)
├── LocateAnything.pdf         # paper
├── LICENSE_MODEL              # NVIDIA model license (code license is repo-root Apache-2.0)
├── assets/                    # teaser/method images for README
├── deepspeed_configs/         # zero_stage1_config.json, zero_stage2_config.json
├── shell/
│   └── locate-anything-streaming.sh   # reference single/multi-node training launcher
├── document/                  # TRAINING / DATA_PREPARATION / STREAMING_PACKING / RESULTS
├── evaluation/                # DDP inference + metrics + scripts (+ vendored fastevaluate)
└── eaglevl/                   # the Python package
    ├── conversation.py        # fastchat-style templates (LLaVA/InternVL legacy; not central)
    ├── dist_utils.py          # init_dist (slurm/pytorch launcher)
    ├── KERNELS.md             # kernel-focused reference (Triton, magi, fp8, fused ops)
    ├── model/
    │   ├── locany/            # TRAINING model: config + custom Qwen2 + masks + tokenizer
    │   └── moon_vit/          # MoonViT vision encoder (modeling_vit.py)
    ├── utils/locany/          # INFERENCE/EXPORT copy: model + generate_utils + processor
    ├── train/                 # training entry point, dataset, args, augmentation, tools, fastseek
    ├── sp_utils/              # sequence/context parallel (Ulysses, ring, magi plumbing)
    └── patch/                 # monkey patches: fused ops, fp8, packing attn, samplers, ckpt
```

---

## 4. Architecture

### 4.1 Config composition

`LocateAnythingConfig` (`eaglevl/model/locany/configuration_locateanything.py:44`) is a composite
config (`is_composition = True`) with sub-configs:

- `vision_config` → `MoonViTConfig` (`model_type='moonvit'`).
- `text_config` → `Qwen2Config` or `Qwen3Config`, chosen from `text_config.architectures[0]` (`:74-83`).
- `image_token_index = 151667` default — the id of the `<IMG_CONTEXT>` token where visual features are scattered.
- `model_type = 'locateanything'`; `auto_map` is injected at export time so HF can find the classes.

`MoonViTConfig` defaults (`configuration_locateanything.py:15-41`, mirrored in
`model/moon_vit/modeling_vit.py:31`): `patch_size=14`, `num_hidden_layers=27`, `hidden_size=1152`,
`num_attention_heads=16`, `intermediate_size=4304`, `merge_kernel_size=(2,2)`,
`init_pos_emb_height/width=64`.

### 4.2 MoonViT vision encoder (`eaglevl/model/moon_vit/modeling_vit.py`)

Native-resolution ViT operating on **packed variable-length** patch sequences (no fixed grid):

- `MoonVisionPatchEmbed` (`:243`): Conv2d patchify + `Learnable2DInterpPosEmb` (`:209`) that
  **bicubic-interpolates** a learned `(64,64,dim)` grid to each image's actual `grid_hw` (`:223-240`).
- `Rope2DPosEmb` (`:287`): 2D rotary embedding precomputed up to `512x512` grid, shared across layers/heads.
- `MoonVitEncoderLayer` (`:400`): pre-norm, QKV-packed attention with 2D RoPE, MLP (`MLP2`, GELU-tanh).
  Attention dispatched through `VL_VISION_ATTENTION_FUNCTIONS` (`:172`):
  `flash_attention_2` (`flash_attn_varlen_func`), `sdpa`, or `eager`. Sequences are packed and
  separated by `cu_seqlens` (cumulative lengths) so images don't attend across each other.
- `patch_merger` (`:533`): merges each non-overlapping `2x2` patch neighborhood by concatenation,
  producing `hidden_size*4 = 4608`-dim tokens and reducing token count 4x. Output is a **list of
  per-image token tensors** (one entry per image).
- The LocateAnything model forces MoonViT to `flash_attention_2` during training
  (`model/locany/modeling_locateanything.py:95`); the inference copy falls back to `sdpa` if
  flash-attn is unavailable (`utils/locany/modeling_locateanything.py:104-110`).

### 4.3 MLP projector

`self.mlp1` (`model/locany/modeling_locateanything.py:121-126`):
`LayerNorm(4608) → Linear(4608 → llm_hidden) → GELU → Linear(llm_hidden → llm_hidden)`.
It consumes the 4x-merged MoonViT tokens directly (no pixel-shuffle-back). Number of visual tokens
per image therefore equals `(grid_h * grid_w) / (merge_h * merge_w)`
(`utils/locany/processing_locateanything.py:395`).

### 4.4 LLM decoder + multimodal fusion

`Qwen2ForCausalLM`/`Qwen3ForCausalLM` (custom Qwen2 in `model/locany/modeling_qwen2.py`).
Fusion (`model/locany/modeling_locateanything.py:197-237`):

1. Embed `input_ids` with the LLM input embeddings.
2. Run MoonViT → MLP projector → flat `vit_embeds`.
3. Scatter visual embeds into positions where `input_ids == image_token_index`
   (`input_embeds[selected] = input_embeds[selected]*0 + vit_embeds`).
4. `image_flags` marks which packed samples actually carry images; text-only samples add a
   zero-weight image term to keep the vision graph connected (`:232-236`, and the `ignore_flag`
   path that zeroes the loss `:307-308`).

### 4.5 Special tokens and coordinate vocabulary

Added to the tokenizer at training time (`eaglevl/train/constants.py`, applied at
`train/locany_finetune_magi_stream.py:1273`):

- Structure: `<IMG_CONTEXT>`, `<img>`, `</img>`, `<box>`, `</box>`, `<quad>`, `</quad>`,
  `<ref>`, `</ref>`, `<interval>`, `</interval>`, `<text_mask>`, `<null>`, `</c>` (category separator).
- Coordinates: `<0> … <1000>` — **1001 coordinate tokens** (`constants.py:36`) representing
  normalized positions in `[0, 1000]`.

Token ids are not hardcoded in the model — the training script resolves them from the tokenizer and
injects them into the config (`train/locany_finetune_magi_stream.py:1279-1314`):
`box_start/box_end`, `coord_start/coord_end`, `ref_start/ref_end`, `none`, plus
`text_config.text_mask_token_id` / `null_token_id`. `generate_utils.get_token_ids_from_config`
reads them back at inference with documented fallback defaults
(`utils/locany/generate_utils.py:15-48`).

---

## 5. Parallel Box Decoding (PBD) — the core mechanism

### 5.1 The idea

- **NTP (slow mode):** a box is `<box><x1><y1><x2><y2></box>`, decoded one token per step.
- **MTP / PBD (fast mode):** an **anchor token** is followed by a fixed-length block of
  `<text_mask>` tokens; the model predicts all positions of the block in **one forward pass**.
  A 4-coordinate box block is decoded atomically. Block length = `block_size`
  (`--block_size`, default 4 in args, **6** in docs/scripts).
- **Hybrid:** MTP by default; if a block looks malformed or spatially ambiguous, discard it and
  fall back to NTP for that block, then switch back to MTP at the next `</box>`.

### 5.2 Output formats (normalized integers in `[0,1000]`)

| Output | Tokens |
|--------|--------|
| Box | `<ref>label</ref><box><x1><y1><x2><y2></box>` |
| Point | `<box><x><y></box>` |
| No object | `<box>none</box>` |

Convert to pixels: `coord / 1000 * (width or height)`. Parsers:
`LocateAnythingWorker.parse_boxes/parse_points` (`locateanything_worker.py:142-169`).

### 5.3 Training-time MTP attention mask

This is the heart of how PBD is *learned* (custom Qwen2 in `model/locany/modeling_qwen2.py`,
masks in `model/locany/mask_sdpa_utils.py` and `mask_magi_utils.py`).

- A sample is laid out as a **causal prefix `x0`** (normal context) followed by one or more **MTP
  blocks** of `<text_mask>` tokens that predict future tokens in parallel.
- The prefix/MTP boundary is detected **purely from `position_ids`**: positions increase through
  `x0`, then **drop** when the first MTP block reuses earlier indices
  (`mask_sdpa_utils.py:232-285`). This is the "block-diff by position embedding" idea — mask
  geometry comes from where the position sequence resets, not from token contents.
- `create_block_diff_mask_by_pe_4d` (`mask_sdpa_utils.py:142`) and the packed variant
  `create_mtp_packing_mask_4d` (`:288`) build the mask as the union of three regions:
  1. **Causal prefix** — `x0` tokens attend causally to earlier `x0`.
  2. **Mutual block** — `<text_mask>` queries/keys in the *same* MTP block attend **bidirectionally**
     when `causal_attn=False` (the default), letting a block jointly predict several tokens.
  3. **Block→prefix** — each block attends back into `x0` up to the prefix length recorded by its
     starting `position_id`.
- **Packing isolation:** in packing mode, `sub_sample_lengths` is converted to a per-token
  `data_index`, and the mask is intersected with a `same_sample` gate so packed samples never
  attend across boundaries (`mask_sdpa_utils.py:377`).
- Config fields that drive this (read defensively via `getattr` and injected by the training
  script): `block_size` (default 4), `causal_attn` (default False), `text_mask_token_id`,
  `is_packing_mode` (a runtime attribute flipped to True at `train/...:1375`).

### 5.4 Magi vs SDPA mask representation

The same mask, two encodings:

- **SDPA:** a dense 4D float mask `[B,1,seq,seq]` (`0.0` allowed / `-inf` masked).
- **Magi:** a sparse **range plan** `{q_ranges, k_ranges, attn_type_map, max_seqlen_q, max_seqlen_k}`
  consumed by `flex_flash_attn_func`. `attn_type_map` encodes `FULL=0` / `CAUSAL=1`
  (`mask_magi_utils.py:20`). `convert_mtp_mask_to_magi_plan` (`:121`) emits per-sample range pairs
  for the three regions; cross-sample isolation is structural (ranges never cross sample
  boundaries), so no `same_sample` gate is needed. `build_magi_ranges` (`:23`) is the
  inference/decode analogue.

### 5.5 Inference generation loop (`utils/locany/modeling_locateanything.py:304-537`)

Batch size must be 1; `use_cache=True` required (`:326-328`). Visual features are extracted once
before the loop (`:336-345`).

- `generation_mode` ∈ `{fast, slow, hybrid}` (`:347-353`); `n_future_tokens` default 6.
- **MTP step** (`_sample_token_in_mtp`, `:413`): appends the last token + `(n_future_tokens-1)`
  `<text_mask>` tokens, fixes position ids (`:373-396`), runs the LLM, then samples the block via
  `sample_tokens` → `decode_bbox_avg` / `decode_ref` (`generate_utils.py`). `decode_bbox_avg` uses
  a top-k weighted vote per coordinate and, in hybrid mode, zeroes "abnormal" coordinates whose
  candidate spread is too large (`generate_utils.py:339-353`).
- `handle_pattern` (`generate_utils.py:408`) classifies the block: `im_end` (terminal),
  `empty_box` (`<box>none</box>`), `coord_box` (4-coord box), `point_box` (2-coord point),
  `error_box` (malformed → `need_switch_to_ar=True`), or `ref_object` (a `<ref>` label).
- **AR step** (`_sample_token_in_ar`, `:430`): single-token decode; in hybrid mode it watches for
  `</box>` to switch back to MTP.
- **Mode switching** (`:499-510`): `error_box` → `use_mtp=False`; `box_end_ar` → `use_mtp=True`;
  `im_end` breaks. `fast` stays MTP forever, `slow` stays AR forever.
- KV cache is truncated to the verified prefix each round (`:483-486`) so discarded MTP blocks
  don't poison the cache.
- `verbose=True` returns `(response, sampling_history, stats)` with a `Statistic Info` line
  (`tps`, `bps`, `num_boxes`, `switch_to_ar`, …) that `evaluation/metrics/analyze_speed.py` parses.

---

## 6. Data Pipeline

### 6.1 Annotation format (JSONL, ShareGPT)

Each line: `{"conversations": [{"from": "human"|"gpt", "value": str}, ...], "image"|"image_list"|"video"|"video_list": ...}`
(see `document/DATA_PREPARATION.md`). Conventions:

- Box coordinates are **normalized integers `[0,1000]`**.
- Image placeholders `<image-1>`, `<image-2>`, … in conversation text; if absent, `<image-1>` is
  prepended automatically.
- Detection prompts separate categories with `</c>`.
- Samples without media are treated as text-only.

### 6.2 Recipe JSON (`--meta_path`)

Maps dataset name → `{annotation (str|list), root, repeat_time, data_augment}`
(`document/DATA_PREPARATION.md`, parsed at `train/locany_finetune_magi_stream.py:1162-1212`).
`repeat_time >= 1` repeats; `< 1` downsamples (deterministic subsample with fixed seed 10086,
`...:234-243`). Sampling weight `= repeat_time * len(ds)` (`:1185`).

### 6.3 Image processor (`utils/locany/image_processing_locateanything.py`)

`LocateAnythingImageProcessor`: `patch_size=14`, `in_token_limit=4096`, `merge_kernel_size=[2,2]`,
normalize mean/std `0.5`. `rescale` (`:46`) shrinks images that exceed the token limit, then pads
H/W up to a multiple of `merge*patch` (28), and **rejects grids ≥ 512** ("Exceed pos emb", `:68`).
Outputs `pixel_values` (flattened patches) + `image_grid_hws`.

### 6.4 Processor (`utils/locany/processing_locateanything.py`)

`LocateAnythingProcessor` ties tokenizer + image processor:

- `py_apply_chat_template` (`:611`) builds the Qwen `<|im_start|>…<|im_end|>` chat string and
  injects `<image-N>` / `<video-N>` placeholders; adds the assistant generation prompt.
- `replace_media_placeholder` (`:375`) expands each `<image-N>` to
  `<image N><img>{<IMG_CONTEXT> * num_tokens}</img>` with the right token count per image.
- `process_vision_info` (`:561`) fetches images/videos. Image loaders support local paths, http,
  base64, `file://`, PIL, and **LMDB** records (`parse_lmdb_image_data`, `:98`; AgiBotWorld special
  case `:81`). Video via decord (preferred) or torchvision, with FPS/frame-count sampling
  (`get_video_frame_indices`, `:161`) and a pixel budget.

### 6.5 MTP label construction (`train/locany_finetune_magi_stream.py`)

`LazySupervisedDatasetMTP.get_targets_flag_with_mtp` (`:250-475`) turns a tokenized sample into the
MTP training layout. Two branches:

- **No `</box>`/`</ref>` present** (`:299-390`): split each assistant response into random
  `block_size`-sized blocks (random offset for reproducibility via per-sample seed). Each block
  becomes `[anchor, <text_mask> * (block_size-1)]` inputs with the true future tokens as targets.
- **Detection sequences present** (`:392-475`): box/ref-aware splitting that ends a block at the
  first `</ref>` or `</box>` so coordinate blocks align to box boundaries.

Both branches append the MTP blocks after the original sequence, set `position_ids` to re-anchor
each block, and pad. `multi_modal_get_item` (`:477`) wires text + pixel_values + `image_flags` +
`image_grid_hws` together. Per-sample seeding (`seed = idx + 10086`, `:520-522`) plus up-to-10
retries on failure (`:524-538`).

### 6.6 Streaming (online) packing

See `document/STREAMING_PACKING.md` for the full algorithm. In code
(`train/locany_finetune_magi_stream.py`):

- `LazyJsonlLoader` (`:133`): offset-indexed, thread-safe, lazy JSONL reader.
- `DeterministicIterator` (`:572`): epoch-shuffled deterministic iteration with `state_dict`.
- `StreamPackedDatasetMTP` (`:668`): `IterableDataset` that packs samples up to `max_num_tokens`
  using **Best-Fit** (largest buffered sample that still fits) + **Big-Rocks-First** (start each
  batch with the largest buffered sample). Buffer size = `--packing_buffer_size` (default 32).
  Samples longer than `--max_num_tokens_per_sample` are skipped (`:877-879`).
- `PackedCollatorMTP` (`:1011`): asserts `batch_size == 1`; emits `sub_sample_lengths` describing
  the packed sub-samples. **Always set `--per_device_train_batch_size 1`** — packing controls the
  effective batch.
- `StateAwareDataLoader` (`:1022`) + `DataloaderStateCallback` (`:1045`): persist per-rank packing
  state to `checkpoint-{step}/dataloader_state_rank{rank}.pt` for **bit-wise identical resume**.

---

## 7. Training

### 7.1 Entry point

`eaglevl/train/locany_finetune_magi_stream.py` (the one referenced everywhere; `train_mem`-style
single file). Flow of `main()`:

1. `init_dist(launcher=...)` — `LAUNCHER` env defaults to `slurm`; the shell script sets
   `LAUNCHER=pytorch` for torchrun (`shell/locate-anything-streaming.sh:39`).
2. Apply patches: `replace_liger_fused_ops()` at import (`:83`), `replace_train_dataloader()`,
   `replace_train_sampler()` (`:1231-1232`). A module-level monkey patch makes HF accept
   `"magi"` as an attn implementation (`:89-97`).
3. Load tokenizer, add special + coordinate tokens, resolve all special token ids (`:1269-1314`).
4. Load `LocateAnythingForConditionalGeneration.from_pretrained` with the chosen
   `attn_implementation` and inject token ids/block_size/causal_attn into the config
   (`:1291-1326`). The scratch path (no `model_name_or_path`) instead loads a MoonViT + LLM and
   composes a fresh config (`:1338-1370`).
5. `model.language_model.model.is_packing_mode = True` (`:1375`).
6. Resize token embeddings for the new tokens and init them with the mean of existing embeddings
   (`:1383-1390`).
7. Freeze flags, optional LoRA wraps, optional sequence parallel (`set_pg_manager`, `:1399-1401`).
8. Build the stream-packed dataset, build `StreamPackingMTPTrainer`, train, save.
9. **Export:** rank 0 copies `utils/locany/*.py` into `output_dir` and writes the `auto_map`
   into `config.json` (`:1534-1561`); writes a `done.txt` sentinel (`:1570`).

### 7.2 Key arguments (`eaglevl/train/arguments.py`)

Model args: `block_size` (4), `causal_attn` (False), `attn_implementation` (`magi`),
`freeze_llm/backbone/mlp`, `use_backbone_lora/use_llm_lora`, `mlp_connector_layers` (2),
`vision_select_layer` (-1), `grad_checkpoint`, `use_fp8`, `lr_scale`.
Data args: `max_seq_length` (2048), `meta_path`, `max_num_tokens_per_sample` (32768),
`max_num_tokens` (36864), `packing_buffer_size` (32), `max_frames` (64), `target_fps` (2),
`video_total_pixels`, `sequence_parallel_degree` (1), `sample_log_interval` (100).

> The `arguments.py` defaults differ from the doc/script examples (e.g. `block_size` 4 vs 6,
> `max_seq_length` 2048 vs 16384). The **executable scripts and `document/TRAINING.md` are the
> source of truth** for real runs.

### 7.3 Attention backends

| `--attn_implementation` | GPU | Max seq | Notes |
|-------------------------|-----|---------|-------|
| `magi` | Hopper/Blackwell only | 32K+ | MagiAttention `flex_flash_attn_func`; the PBD training default. |
| `sdpa` | any | ~4K | PyTorch SDPA; the portable fallback. |

Dispatch & fallback in the custom Qwen2 stack: `magi → flash_attention_2 → sdpa` when libraries are
missing (`model/locany/modeling_qwen2.py`, summarized in `KERNELS.md:48`). **`flash_attention_2`
cannot run MTP packing** — the mask path raises `NotImplementedError` for `<text_mask>`/4D masks, so
do not use it for PBD training. MagiAttention only supports Hopper/Blackwell; on A100/L40 use `sdpa`
and keep sequences short.

### 7.4 DeepSpeed & parallelism

`deepspeed_configs/zero_stage1_config.json` (lower comms) and `zero_stage2_config.json`
(recommended). Both leave lr/batch/clip as `"auto"` (inherit from Trainer). Sequence/context
parallel utilities live in `eaglevl/sp_utils/` (Ulysses all-to-all + ring variants), enabled when
`sequence_parallel_degree > 1`.

### 7.5 Resume

Resume is automatic when `output_dir` has a checkpoint. The script sets `ignore_data_skip=True`
and loads `dataloader_state_rank{rank}.pt` to restore packing buffers/iterators/RNG for
deterministic continuation (`:1508-1529`). If the recipe JSON changed since the checkpoint, resume
determinism is broken.

### 7.6 Launch (reference)

`shell/locate-anything-streaming.sh` is a runnable single/multi-node template (`GPUS`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `OUTPUT_DIR`, `MODEL_PATH`). Note its small debug values
(`--max_seq_length 6400`, `--meta_path ./recipe/ablation.json`) are placeholders — replace before a
real run.

---

## 8. Inference (Worker API)

`locateanything_worker.py` is the user-facing entry. `LocateAnythingWorker(model_path)` loads
tokenizer/processor/model once (`trust_remote_code=True`, bf16, `.eval()`), then:

- `predict(image, question, generation_mode="hybrid", max_new_tokens=2048, temperature=0.7, verbose=True)`
  builds chat messages, runs `processor.process_vision_info` + `model.generate(...)`, returns
  `{"answer", "history"?, "stats"?}` (`:34-97`).
- Task helpers wrap the right prompt string: `detect` (categories joined by `</c>`),
  `ground_single` / `ground_multi`, `ground_text`, `detect_text`, `ground_gui(output_type=box|point)`,
  `point` (`:101-138`).
- `parse_boxes` / `parse_points` convert `[0,1000]` token coords to pixels (`:142-169`).

Generation defaults passed through: `do_sample=True, top_p=0.9, repetition_penalty=1.1`,
`use_cache=True` (`:77-91`).

---

## 9. Evaluation (`evaluation/`)

Three categories, each a 3–4 stage pipeline: **DDP inference → (convert) → metric → speed**.

| Category | Script | Inference | Pred file | Metric |
|----------|--------|-----------|-----------|--------|
| COCO/LVIS detection (box) | `eval_coco.sh`, `eval_lvis.sh` | `inference_detection_ddp.py` | `eval_results.jsonl` → `fast_eval.tsv` | `metrics/coco_lvis_metric.py` (Rex-Omni `fastevaluate`) |
| Other grounding (box/point) | `eval_grounding.sh` | `inference_grounding_ddp.py` | `answer.jsonl` | `metrics/other_metric.py` |
| ScreenSpot-Pro (GUI box) | `eval_sspro.sh` | `inference_sspro_ddp.py` | `predictions.jsonl` | `metrics/sspro_metric.py` |

Mechanics:

- **Launch:** `torchrun --nproc_per_node=$GPUS` (defaults `GPUS=8`, NCCL, 2h timeout). Auto-clamps
  to available GPUs and hard-exits if `nvidia-smi` is missing.
- **Sharding/merge:** `DistributedSampler(shuffle=False)`, `batch_size=1`; each rank writes
  `<name>_rank<r>.jsonl`, then `dist.barrier()`, then **rank 0 concatenates** into the final pred
  file and deletes temp shards. (`evaluation/utils/merge_pred.py` is a standalone manual fallback,
  not wired into the scripts.)
- **Model API:** each inference script defines its **own in-file `LocateAnythingWorker`** (not the
  top-level one), loads `AutoModel`/`AutoProcessor` with `trust_remote_code=True`, and calls
  `model.generate(...)` directly. `generation_mode` defaults to `hybrid`; `fast`/`hybrid` add
  `n_future_tokens=6`. Shared kwargs/decoding via `evaluation/inference_compat.py`.
- **Prompts** vary by dataset/eval_type (`inference_grounding_ddp.py`): `point_eval` → "Point to:";
  RefCOCOg → single instance; HumanRef → all instances; OCR set `{TotalText,SROIE,IC15,HierText}`
  → "Detect all the text in box format."; COCO/LVIS/VisDrone point_eval issue **one query per
  category** (expensive, separate path).
- **`eval_grounding.sh` options:** `--eval_type box_eval|point_eval` and `--dataset` from a broad
  case list (`LVIS, COCO, HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc, RefCOCOg_test,
  RefCOCOg_val, SROIE, TotalText, VisDrone, FSCD_test`). The JSONL path is auto-derived as
  `${IMAGE_ROOT}/_annotations/${EVAL_TYPE}/<Dataset>.jsonl` unless `--test_jsonl` overrides.
- **Metrics:** COCO/LVIS → AP/AR/F1 (overall + @0.50/@0.95); grounding → per-task table with
  IoU 0.5/0.9 + mIoU mean F1, MAE, pointing/keypoint/hallucination/GUI accuracies; sspro →
  point-in-GT-box accuracy. `analyze_speed.py` only aggregates the `Statistic Info` lines the model
  prints (TPS/BPS), it doesn't compute throughput itself.

Setup gotchas (`evaluation/README.md`): install Rex-Omni `fastevaluate` (vendored under
`evaluation/fastevaluate/` but must be `pip install -e .`) **plus** `shapely`; download
`Rex-Omni-EvalData` and **untar the `*.tar.gz` archives in place**; ScreenSpot-Pro needs a separate
`converted_box.jsonl` (used as both the test jsonl and the `--type_match_file`) and a different
`--image_root` (`ScreenSpot-Pro/images`). The `_annotations/<eval_type>/<Dataset>.jsonl` subtree
must exist exactly.

---

## 10. Kernels, Fused Ops, and Sequence Parallel

See `eaglevl/KERNELS.md` for the authoritative kernel map. Highlights:

- **Fused ops** (`patch/fused_ops/`): Triton RMSNorm, SwiGLU, RoPE (Liger/Unsloth-derived).
  `replace_liger_fused_ops()` swaps HF Qwen2/Qwen3/Llama MLP+RMSNorm (and Qwen3 RoPE) at import.
- **Loss:** `train/liger_loss_weight_ops.py` provides `LigerFusedLinearCrossEntropyLoss` used by the
  training model (avoids materializing full logits).
- **FP8:** `patch/fp8_gemm.py` wraps external `deep_gemm` for FP8 GEMM (`--use_fp8`); skips vision,
  attention, even layers, and `lm_head`.
- **Packing attention:** `patch/packing_attention.py` monkey-patches HF FlashAttention to preserve
  integer packing masks (not used by the MTP path, which has its own masks).
- **Sequence parallel** (`sp_utils/`): Ulysses all-to-all (`attention.py`, `all_to_all.py`,
  `ulysses_attn.py`), ring/zigzag/stripe variants (`sp_utils/ring/`), `globals.py` PG manager,
  `monkey_patch.py`. Packing SP support is Ulysses-focused (ring packing is not guaranteed).
- External native kernels come from `flash_attn`, `magi_attention`, `deep_gemm`, `liger_kernel`,
  optional `apex`. There are **no in-repo `.cu` files**.

---

## 11. Gotchas Checklist

- **Two copies:** edit the right one. PBD generation + processor live in `utils/locany/`; training
  model + masks live in `model/locany/`. Mirror cross-cutting changes.
- **Don't change output token formats** (`<ref>/<box>/</c>`, `<0>…<1000>`) without updating training
  label construction, the inference parser (`handle_pattern`, `parse_boxes`), and every evaluation
  parser.
- **`per_device_train_batch_size` must be 1** — streaming packing controls the effective batch.
- **`magi` only on Hopper/Blackwell**; use `sdpa` (≤~4K) elsewhere. `flash_attention_2` cannot run
  MTP packing.
- **Pinned deps are coupled** (`transformers==4.57.1`, `tokenizers==0.22.0`, `deepspeed==0.15.4`,
  `liger_kernel==0.3.1`). Don't upgrade casually; remote-code checkpoints depend on exact versions.
- **Placeholders everywhere** in scripts (`path/to/...`, `./recipe/ablation.json`,
  `local_playground/...`). Replace before running.
- **Resume needs an unchanged recipe** and the per-rank `dataloader_state_rank*.pt` files.
- **Trained checkpoints become loadable** only after the post-train export step writes `auto_map`
  and copies `utils/locany/*.py`; don't strip `trust_remote_code=True`.
- **Don't commit** `work_dirs*`, checkpoints, `training_log.txt`, `done.txt`, extracted EvalData,
  parquet/LMDB, or TensorRT/W&B outputs.
- **Most validation needs GPUs, exact external kernels, and prepared data.** For docs-only edits run
  `git diff --check`; for Python syntax `python -m py_compile <file>`.
