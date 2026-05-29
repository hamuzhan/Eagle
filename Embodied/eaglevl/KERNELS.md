# LocateAnything eaglevl Kernel Map

This file is a kernel-focused reference for `Embodied/eaglevl` only. It is not an OpenCode auto-loaded instruction file.

Do not confuse this package with `Eagle2_5/eaglevl`: both import as `eaglevl` and share some Triton files, but `Embodied/eaglevl` adds LocateAnything-specific `locany`, `moon_vit`, MagiAttention wiring, and `patch/fp8_gemm.py`.

## Kernel-Relevant Layout

- `model/locany/`: training/runtime LocateAnything model code. `modeling_qwen2.py` owns the LLM attention classes, including MagiAttention; `modeling_locateanything.py` wires MoonViT, Qwen2/Qwen3, sequence parallel helpers, and fused loss.
- `model/moon_vit/`: MoonViT vision encoder. Its attention dispatch uses FlashAttention varlen, SDPA, or eager attention.
- `utils/locany/`: Hugging Face `trust_remote_code` export copy used by released checkpoints. It mirrors much of `model/locany/` plus processor/image-processor files; attention or mask changes usually need to be mirrored here.
- `patch/fused_ops/`: in-repo Triton fused RMSNorm, SwiGLU, RoPE, and shared helpers.
- `patch/fp8_gemm.py`: Embodied-only FP8 quantization wrappers around external `deep_gemm` kernels.
- `patch/packing_attention.py`: Hugging Face FlashAttention packing monkey patch, including transformer-version-specific mask handling.
- `sp_utils/`: sequence-parallel utilities. Ulysses/all-to-all code is in `attention.py`, `all_to_all.py`, and `ulysses_attn.py`; ring variants live in `sp_utils/ring/`.
- `train/liger_loss_weight_ops.py`: fused linear cross-entropy loss with local Triton scaling and external `liger_kernel` cross entropy.

## In-Repo Triton Kernels

- `patch/fused_ops/fused_rms_norm.py:46` `_rms_norm_forward_kernel`: row-wise RMSNorm forward. It computes and caches per-row `RSTD`, supports `llama`, `gemma`, and `none` casting modes, and is wrapped by `rms_norm_forward` and `LigerRMSNormFunction`.
- `patch/fused_ops/fused_rms_norm.py:118` `_rms_norm_backward_kernel`: computes `dX` and partial `dW`; wrapper reduction is in `rms_norm_backward`. Preserve the Unsloth/Liger attribution header if editing this file.
- `patch/fused_ops/fused_rms_norm.py:376` `LigerRMSNorm`: module used by `replace_liger_fused_ops()` to replace HF Qwen/Llama RMSNorm classes.
- `patch/fused_ops/fused_swiglu.py:10` `silu`: Triton helper used inside the SwiGLU forward kernel.
- `patch/fused_ops/fused_swiglu.py:15` `_swiglu_forward_kernel`: computes `silu(a) * b` row-wise; launched by `swiglu_forward`.
- `patch/fused_ops/fused_swiglu.py:36` `_swiglu_backward_kernel`: recomputes SiLU during backward to save memory; writes gradients back through the saved `a` and `b` buffers.
- `patch/fused_ops/fused_swiglu.py:108` `LigerSiLUMulFunction` and `:124` `LigerSwiGLUMLP`: autograd/module wrappers used as HF MLP replacements.
- `patch/fused_ops/fused_rotary_pos_emb.py:8` `_triton_rope`: one Triton RoPE kernel for forward and backward, switched by `BACKWARD_PASS`; implements the HF Llama/Mistral RoPE layout.
- `patch/fused_ops/fused_rotary_pos_emb.py:207` `LigerRopeFunction` and `:244` `liger_rotary_pos_emb`: autograd/function wrappers. `replace_liger_fused_ops()` currently applies this only to HF Qwen3 RoPE; Qwen2 RoPE replacement is commented out with a TODO.
- `patch/fused_ops/utils.py:52` `calculate_settings`: picks block size/warps for fused kernels; do not hardcode launch sizes elsewhere without checking this helper.
- `patch/fused_ops/utils.py:103` `element_mul_kernel`: small in-place row scaling kernel shared by fused-loss style code.
- `patch/fp8_gemm.py:14` `_per_token_cast_to_fp8_kernel`: per-token BF16 to FP8 E4M3 cast and scale generation for activations.
- `patch/fp8_gemm.py:62` `_per_block_cast_to_fp8_kernel`: per-block BF16 to FP8 E4M3 cast and scale generation for weights.
- `patch/fp8_gemm.py:198` `_copy_kernel`: row copy/remap kernel used by grouped FP8 GEMM when padding split groups.
- `patch/fp8_gemm.py:144` `deep_matmul` and `:222` `deep_group_matmul`: call external `deep_gemm`; this file does not implement the actual GEMM kernel in Triton.
- `patch/fp8_gemm.py:267` `_DeepLinear`, `:294` `DeepLinear`, and `:303` `apply_fp8`: monkey-patch selected `torch.nn.Linear.forward` methods. It skips paths containing `vision` or `self_attn`, skips even LLM layers, and warns not to use FP8 for `lm_head`.
- `sp_utils/ring/triton_utils.py:7` `flatten_kernel` and `:39` `flatten_varlen_lse`: flatten batched FlashAttention LSE into varlen/ring layout.
- `sp_utils/ring/triton_utils.py:71` `unflatten_kernel` and `:103` `unflatten_varlen_lse`: restore flattened LSE to batched layout.
- `train/liger_loss_weight_ops.py:53` `element_mul_kernel`: local loss-weight scaling kernel, separate from `patch/fused_ops/utils.py`.
- `train/liger_loss_weight_ops.py:87` `fused_linear_cross_entropy_forward`: chunks the linear projection and launches external `liger_cross_entropy_kernel` at `:164`; backward is `:264`, wrapper function is `:314`, module is `LigerFusedLinearCrossEntropyLoss` at `:366`.

There are no model-owned `.cu` or `.cuh` CUDA kernel files under `Embodied/eaglevl`. Native kernels used by LocateAnything come from external packages such as `flash_attn`, `magi_attention`, `deep_gemm`, `liger_kernel`, or optional `apex`.

## Attention Backends

- `model/locany/modeling_qwen2.py:52` guards MagiAttention import from `magi_attention.functional.flex_flash_attn`; missing Magi falls back later.
- `model/locany/modeling_qwen2.py:769` `Qwen2MagiAttention` calls external `flex_flash_attn_func` at `:849` with `q_ranges`, `k_ranges`, and `attn_type_map`.
- `model/locany/modeling_qwen2.py:870` `QWEN2_ATTENTION_CLASSES` maps `eager`, `flash_attention_2`, `sdpa`, and `magi`.
- `model/locany/modeling_qwen2.py:884` fallback order is `magi` to `flash_attention_2` when available, otherwise `sdpa`; missing FlashAttention also falls back to `sdpa` at `:895`.
- `train/arguments.py:111` defaults `attn_implementation` to `magi`; valid values are `magi`, `flash_attention_2`, `sdpa`, and `eager`.
- `model/locany/mask_magi_utils.py:23` `build_magi_ranges` builds inference/decode Magi plans; `:122` `convert_mtp_mask_to_magi_plan` converts packed MTP masks. `FULL` and `CAUSAL` are encoded as `0` and `1` at `:20`.
- `model/locany/mask_sdpa_utils.py:142` `create_block_diff_mask_by_pe_4d` builds block-difference SDPA masks for non-packed data; `:289` `create_mtp_packing_mask_4d` handles stream packing.
- `model/moon_vit/modeling_vit.py:60` `multihead_attention` uses external `flash_attn_varlen_func`; `:109` `sdpa_attention` uses PyTorch `scaled_dot_product_attention`; `:141` `eager_attention` is the explicit matmul-softmax path.
- `model/moon_vit/modeling_vit.py:172` `VL_VISION_ATTENTION_FUNCTIONS` dispatches MoonViT attention backends. `model/locany/modeling_locateanything.py:95` forces `config.vision_config._attn_implementation = 'flash_attention_2'` for MoonViT.
- `patch/packing_attention.py:216` `flash_attention_forward_for_packing` preserves integer packing masks and calls the local `_flash_attention_forward` at `:44`.
- `patch/packing_attention.py:309` `patch_packing_attention()` patches HF `AttentionInterface['flash_attention_2']` at `:317`. For transformers >= 4.55 it patches `_preprocess_mask_arguments` at `:328` so integer packing masks are not converted to bool.

## Sequence Parallel Attention

- `sp_utils/attention.py` performs Ulysses-style all-to-all preprocessing and postprocessing for attention; it relies on `get_pg_manager().ulysses_sequence_parallel_world_size`.
- `sp_utils/ring/__init__.py` exports ring, stripe, zigzag ring, and varlen ring FlashAttention wrappers. These are adapted from `ring-flash-attention`; keep attribution comments.
- `sp_utils/ring/triton_utils.py` only reshapes LSE. The actual attention math still comes from FlashAttention-style kernels called by the ring wrapper files.
- `patch/packing_attention.py` has a TODO noting current packing support is Ulysses-focused; do not assume ring packing is fully supported without testing the specific path.

## Kernel Activation Paths

- `patch/fused_monkey_patch.py:64` `replace_liger_fused_ops()` monkey-patches HF Qwen2/Llama/Qwen3 MLP and RMSNorm classes to `LigerSwiGLUMLP` and `LigerRMSNorm`.
- `patch/fused_monkey_patch.py:82` also replaces HF Qwen3 `apply_rotary_pos_emb` with `liger_rotary_pos_emb`; Qwen2 RoPE replacement is intentionally commented out at `:73`.
- `patch/fused_monkey_patch.py:10` `FusedSiglipMLP` wraps `flash_attn.ops.fused_dense.fused_mlp_func`; `replace_siglip_fused_ops()` at `:55` swaps SigLIP MLP only if that import succeeded.
- `patch/llm_rmsnorm_monkey_patch.py:4` `replace_llm_rmsnorm_with_fused_rmsnorm()` optionally swaps Llama/Qwen2 RMSNorm to `apex.normalization.FusedRMSNorm`; it silently falls back if Apex is absent or broken.
- `model/locany/modeling_locateanything.py:296` uses `LigerFusedLinearCrossEntropyLoss` for language-model loss.
- These patches are opt-in. Importing `eaglevl.patch` does not by itself replace model classes; find the caller before assuming a fused kernel is active.

## External Kernels And Hardware Constraints

- `flash_attn` supplies FlashAttention varlen and dense kernels for Qwen/MoonViT and fused dense MLP. Version behavior matters; FlashAttention < 2.1 used top-left causal masks in some paths.
- `magi_attention` supplies `flex_flash_attn_func`, the real MagiAttention kernel. LocateAnything training defaults to `magi`, but MagiAttention is intended for Hopper/Blackwell GPUs; use `sdpa` on unsupported GPUs.
- `deep_gemm` supplies FP8 GEMM in `patch/fp8_gemm.py`; the in-repo Triton code only prepares FP8 tensors and copies padded rows.
- `liger_kernel` supplies the cross-entropy kernel used by `train/liger_loss_weight_ops.py`.
- `apex` is optional and only used by the RMSNorm monkey patch if available.
- Do not casually upgrade `torch`, `triton`, `transformers`, `flash-attn`, `deep_gemm`, or `liger_kernel`; this repo pins dependency stacks per subproject.

## Editing Gotchas

- Mirror LocateAnything attention, mask, and model changes between `model/locany/` and `utils/locany/` unless the change is intentionally training-only or checkpoint-export-only.
- Preserve NVIDIA, Unsloth, Liger, and ring-flash-attention attribution blocks in kernel-derived files.
- If editing `patch/fp8_gemm.py`, verify the skip rules in `apply_fp8()` still avoid vision modules, attention modules, even LLM layers, and `lm_head`.
- If editing packed attention, test both transformers <= 4.43 and newer `AttentionInterface` paths when possible; the file has version-specific patching.
- Most meaningful validation requires CUDA, exact external kernels, and model paths. For docs-only edits run `git diff --check`; for Python-only syntax checks, `python -m py_compile <touched_file>` is safe but does not validate kernel runtime behavior.
