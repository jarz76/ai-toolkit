# AI-Toolkit Hidden Settings Guide

This document lists all configuration options available in JSON/YAML config mode that are **not exposed in the UI**. Use these settings in your `.yaml` or `.json` config files for advanced control.

---

## Table of Contents
- [Optimizer Options](#optimizer-options)
- [Learning Rate Scheduler Options](#learning-rate-scheduler-options)
- [Train Config (Hidden Settings)](#train-config-hidden-settings)
- [Network Config (Hidden Settings)](#network-config-hidden-settings)
- [Model Config (Hidden Settings)](#model-config-hidden-settings)
- [Dataset Config (Hidden Settings)](#dataset-config-hidden-settings)
- [Sample Config (Hidden Settings)](#sample-config-hidden-settings)
- [Save Config (Hidden Settings)](#save-config-hidden-settings)
- [EMA Config (Hidden Settings)](#ema-config-hidden-settings)
- [Logging Config](#logging-config)

---

## Optimizer Options

The UI only exposes `adamw8bit` and `adafactor`. All available optimizers:

| Optimizer | Description | Notes |
|-----------|-------------|-------|
| `adamw8bit` | AdamW 8-bit (bitsandbytes) | Default, memory efficient |
| `adam8bit` | Adam 8-bit (bitsandbytes) | Without weight decay |
| `ademamix8bit` | AdEMAMix 8-bit | Experimental |
| `adamw` | Standard AdamW (FP32) | Higher VRAM |
| `adam` | Standard Adam (FP32) | Higher VRAM |
| `prodigy` | Prodigy optimizer | Adaptive LR, set `lr: 1.0` |
| `prodigy8bit` | Prodigy 8-bit | Memory efficient Prodigy |
| `lion` | Lion optimizer | `pip install lion-pytorch` |
| `lion8bit` | Lion 8-bit | Memory efficient Lion |
| `dadaptation` | D-Adaptation Adam | Adaptive LR, set `lr: 1.0` |
| `dadaptationlion` | D-Adaptation Lion | Adaptive LR |
| `adafactor` | Adafactor | Memory efficient |
| `adagrad` | Adagrad | Traditional |
| `automagic` | Per-param adaptive LR | Experimental |

---

### Prodigy / Prodigy8bit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1.0` | Learning rate multiplier (keep at 1.0) |
| `betas` | `[0.9, 0.999]` | Coefficients for gradient averages |
| `beta3` | `sqrt(beta2)` | Coefficient for stepsize calculation |
| `eps` | `1e-8` | Numerical stability term |
| `weight_decay` | `0` | L2 regularization |
| `decouple` | `true` | AdamW-style decoupled decay |
| `use_bias_correction` | `false` | Enable Adam bias correction |
| `safeguard_warmup` | `false` | Remove LR from D estimate denominator (warmup safety) |
| `d0` | `1e-6` | Initial D estimate |
| `d_coef` | `1.0` | D coefficient (tune between 0.5-2.0) |
| `growth_rate` | `inf` | Max D growth per step (1.02 = warmup effect) |

```yaml
train:
  optimizer: "prodigy"
  lr: 1.0
  optimizer_params:
    weight_decay: 0.01
    d_coef: 1.0
    use_bias_correction: true
    safeguard_warmup: true
    growth_rate: 1.02  # Acts like warmup
    betas: [0.9, 0.999]
```

---

### Adafactor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `null` | External LR (set for manual LR) |
| `eps` | `[1e-30, 1e-3]` | Regularization constants |
| `clip_threshold` | `1.0` | Gradient update RMS threshold |
| `decay_rate` | `-0.8` | Square average decay coefficient |
| `beta1` | `null` | First moment coefficient (null = disabled) |
| `weight_decay` | `0.0` | L2 penalty |
| `scale_parameter` | `false`* | Scale LR by parameter RMS |
| `relative_step` | `false`* | Time-dependent LR (incompatible with manual LR) |
| `warmup_init` | `false`* | Internal warmup (requires relative_step) |

*Auto-set to false in ai-toolkit for manual LR control

```yaml
train:
  optimizer: "adafactor"
  lr: 1e-4
  optimizer_params:
    weight_decay: 0.01
    clip_threshold: 1.0
    decay_rate: -0.8
    beta1: 0.9  # Enable first moment (optional)
```

---

### Adam8bit / AdamW8bit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betas` | `[0.9, 0.999]` | Coefficients for gradient averages |
| `eps` | `1e-8` | Numerical stability |
| `weight_decay` | `0` | Weight decay coefficient |
| `decouple` | `true` | AdamW-style decoupled decay |

```yaml
train:
  optimizer: "adamw8bit"
  lr: 1e-4
  optimizer_params:
    weight_decay: 1e-4
    betas: [0.9, 0.999]
```

---

### Lion / Lion8bit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betas` | `[0.9, 0.99]` | Coefficients for EMA |
| `weight_decay` | `0` | Weight decay |

```yaml
train:
  optimizer: "lion8bit"
  lr: 1e-5  # Lion uses lower LR than Adam
  optimizer_params:
    weight_decay: 0.01
    betas: [0.9, 0.99]
```

---

### D-Adaptation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1.0` | LR multiplier (keep at 1.0) |
| `eps` | `1e-6` | Numerical stability |
| `weight_decay` | `0` | Weight decay |

```yaml
train:
  optimizer: "dadaptation"
  lr: 1.0
  optimizer_params:
    weight_decay: 0.01
```

---

### Automagic Parameters (Experimental)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-6` | Starting LR |
| `min_lr` | `1e-7` | Minimum per-param LR |
| `max_lr` | `1e-3` | Maximum per-param LR |
| `lr_bump` | `1e-6` | LR adjustment amount |
| `beta2` | `0.999` | Second moment coefficient |
| `clip_threshold` | `1.0` | Gradient clipping |
| `weight_decay` | `0.0` | Weight decay |

```yaml
train:
  optimizer: "automagic"
  lr: 1e-6
  optimizer_params:
    min_lr: 1e-7
    max_lr: 1e-3
    weight_decay: 0.01
```

---

## Learning Rate Scheduler Options

UI defaults to `constant`. Available schedulers:

| Scheduler | Description |
|-----------|-------------|
| `constant` | Constant LR (default) |
| `cosine` | Cosine annealing to minimum |
| `cosine_with_restarts` | Cosine with warm restarts |
| `linear` | Linear decay |
| `step` | Step-wise decay |
| `constant_with_warmup` | Constant with initial warmup |

### Scheduler Parameters

#### constant
```yaml
train:
  lr_scheduler: "constant"
  lr_scheduler_params:
    factor: 1.0  # Multiplier (1.0 = no change)
```

#### cosine
```yaml
train:
  lr_scheduler: "cosine"
  lr_scheduler_params:
    eta_min: 1e-7  # Minimum LR at end
```

#### cosine_with_restarts
```yaml
train:
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    T_mult: 1     # Cycle length multiplier after each restart
    eta_min: 1e-7
```

#### linear
```yaml
train:
  lr_scheduler: "linear"
  lr_scheduler_params:
    start_factor: 1.0
    end_factor: 0.1
```

#### step
```yaml
train:
  lr_scheduler: "step"
  lr_scheduler_params:
    step_size: 500   # Steps between decay
    gamma: 0.5       # Decay multiplier
```

#### constant_with_warmup
```yaml
train:
  lr_scheduler: "constant_with_warmup"
  lr_scheduler_params:
    num_warmup_steps: 500
```

---

## Train Config (Hidden Settings)

### Timestep & Noise Control

| Setting | Default | Description |
|---------|---------|-------------|
| `timestep_type` | `sigmoid` | Options: `sigmoid`, `linear`, `shift`, `weighted`, `lognorm_blend`, `next_sample`, `one_step` |
| `content_or_style` | `balanced` | Timestep bias: `balanced`, `content` (high noise), `style` (low noise) |
| `content_or_style_reg` | `balanced` | Same for regularization images |
| `min_denoising_steps` | `0` | Minimum timestep index |
| `max_denoising_steps` | `999` | Maximum timestep index |
| `num_train_timesteps` | `1000` | Total timesteps |
| `noise_offset` | `0.0` | Noise offset for better contrast |
| `dynamic_noise_offset` | `false` | Adaptive noise offset |
| `noise_multiplier` | `1.0` | Noise intensity scale |
| `random_noise_multiplier` | `0.0` | Random per-sample noise scale |
| `optimal_noise_pairing_samples` | `1` | Samples for optimal noise pairing |
| `force_consistent_noise` | `false` | Deterministic noise per image |
| `blended_blur_noise` | `false` | Blend noise with blurred version |

```yaml
train:
  timestep_type: "sigmoid"
  content_or_style: "balanced"
  min_denoising_steps: 0
  max_denoising_steps: 999
  noise_offset: 0.0357
  force_consistent_noise: false
```

---

### Loss & SNR

| Setting | Default | Description |
|---------|---------|-------------|
| `loss_type` | `mse` | Options: `mse`, `mae`, `wavelet`, `pixelspace`, `mean_flow`, `stepped` |
| `loss_target` | `noise` | Options: `noise`, `source`, `unaugmented`, `differential_noise` |
| `min_snr_gamma` | `null` | Min-SNR weighting (try 5.0) |
| `snr_gamma` | `null` | Fixed SNR gamma |
| `learnable_snr_gos` | `false` | Learnable SNR parameters |
| `match_noise_norm` | `false` | Match noise norm before loss |

```yaml
train:
  loss_type: "mse"
  loss_target: "noise"
  min_snr_gamma: 5.0
```

---

### Regularization & Preservation

| Setting | Default | Description |
|---------|---------|-------------|
| `diff_output_preservation` | `false` | DOP - regularize toward untriggered output |
| `diff_output_preservation_multiplier` | `1.0` | DOP loss weight |
| `diff_output_preservation_class` | `""` | Class word (e.g., "woman") |
| `blank_prompt_preservation` | `false` | Preserve blank prompt behavior |
| `blank_prompt_preservation_multiplier` | `1.0` | BPP loss weight |
| `inverted_mask_prior` | `false` | Regularize unmasked regions |
| `inverted_mask_prior_multiplier` | `0.5` | Masked prior weight |
| `reg_weight` | `1.0` | Reg image loss weight |

```yaml
train:
  diff_output_preservation: true
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_class: "person"
```

---

### CFG & Guidance

| Setting | Default | Description |
|---------|---------|-------------|
| `do_cfg` | `false` | Enable CFG during training |
| `do_random_cfg` | `false` | Random CFG scale |
| `cfg_scale` | `1.0` | CFG scale |
| `max_cfg_scale` | (cfg_scale) | Max for random CFG |
| `bypass_guidance_embedding` | `false` | Bypass guidance (FLUX) |
| `do_guidance_loss` | `false` | Contrastive guidance loss |
| `guidance_loss_target` | `3.0` | Target guidance scale |
| `do_differential_guidance` | `false` | Differential guidance |
| `differential_guidance_scale` | `3.0` | Scale for diff guidance |

```yaml
train:
  bypass_guidance_embedding: true  # For FLUX
  do_cfg: false
  do_guidance_loss: false
```

---

### Memory & Performance

| Setting | Default | Description |
|---------|---------|-------------|
| `xformers` | `false` | Use xformers attention |
| `sdp` | `false` | Use torch SDP attention |
| `attention_backend` | `native` | Options: `native`, `flash`, `_flash_3_hub`, `_flash_3` |
| `gradient_checkpointing` | `true` | Enable gradient checkpointing |
| `unload_text_encoder` | `false` | Unload TE after encoding |
| `cache_text_embeddings` | `false` | Cache text embeddings |
| `do_paramiter_swapping` | `false` | Parameter swapping |
| `paramiter_swapping_factor` | `0.1` | Fraction of params active |

```yaml
train:
  gradient_checkpointing: true
  unload_text_encoder: true
  cache_text_embeddings: true
  attention_backend: "native"
```

---

### Advanced Training

| Setting | Default | Description |
|---------|---------|-------------|
| `max_grad_norm` | `1.0` | Gradient clipping |
| `weight_jitter` | `0.0` | Random weight perturbation |
| `pred_scaler` | `1.0` | Scale prediction |
| `standardize_images` | `false` | Normalize inputs |
| `standardize_latents` | `false` | Normalize latents |
| `correct_pred_norm` | `false` | Correct prediction norm |
| `prompt_dropout_prob` | `0.0` | Per-encoder dropout |
| `prompt_saturation_chance` | `0.0` | Repeat prompt to 77 tokens |
| `short_and_long_captions` | `false` | Dual caption training |
| `single_item_batching` | `false` | Process one at a time |
| `merge_network_on_save` | `false` | Merge LoRA on save |
| `do_prior_divergence` | `false` | Negative loss on prior |

```yaml
train:
  max_grad_norm: 1.0
  weight_jitter: 0.0
  prompt_dropout_prob: 0.05
  single_item_batching: false
```

---

### Component Learning Rates

| Setting | Default | Description |
|---------|---------|-------------|
| `unet_lr` | (lr) | UNet/transformer LR |
| `text_encoder_lr` | (lr) | Text encoder LR |
| `refiner_lr` | (lr) | Refiner LR |
| `embedding_lr` | (lr) | Embedding LR |
| `adapter_lr` | (lr) | Adapter LR |

```yaml
train:
  lr: 1e-4
  unet_lr: 1e-4
  text_encoder_lr: 5e-6
```

---

### Training Targets

| Setting | Default | Description |
|---------|---------|-------------|
| `train_unet` | `true` | Train UNet/transformer |
| `train_text_encoder` | `false` | Train text encoder |
| `train_refiner` | `true` | Train refiner |
| `train_turbo` | `false` | Turbo training mode |

```yaml
train:
  train_unet: true
  train_text_encoder: false
```

---

## Network Config (Hidden Settings)

| Setting | Default | Description |
|---------|---------|-------------|
| `type` | `lora` | Options: `lora`, `locon`, `lokr`, `lorm` |
| `linear` | `4` | Linear layer rank |
| `linear_alpha` | (alpha) | Linear alpha |
| `conv` | `null` | Conv layer rank |
| `conv_alpha` | (conv) | Conv alpha |
| `dropout` | `null` | LoRA dropout rate |
| `transformer_only` | `true` | Only target transformer |
| `lokr_full_rank` | `false` | Full rank LoKR |
| `lokr_factor` | `-1` | LoKR factor (-1 = auto) |
| `old_lokr_format` | `false` | Legacy format |
| `split_multistage_loras` | `true` | Split for multistage |
| `pretrained_lora_path` | `null` | Resume from LoRA |

```yaml
network:
  type: "lora"
  linear: 32
  linear_alpha: 32
  conv: 16
  conv_alpha: 16
  dropout: 0.05
  transformer_only: true
  network_kwargs:
    ignore_if_contains: ["some_layer"]
```

---

## Model Config (Hidden Settings)

| Setting | Default | Description |
|---------|---------|-------------|
| `arch` | (auto) | Model architecture |
| `quantize` | `false` | Quantize transformer |
| `qtype` | `qfloat8` | Quant type: `qfloat8`, `qint8`, `qint4`, `nf4`, `fp4` |
| `quantize_te` | (quantize) | Quantize text encoder |
| `qtype_te` | `qfloat8` | TE quant type |
| `low_vram` | `false` | Low VRAM mode |
| `layer_offloading` | `false` | Layer offloading |
| `layer_offloading_transformer_percent` | `1.0` | % of transformer to offload (0-1) |
| `layer_offloading_text_encoder_percent` | `1.0` | % of TE to offload (0-1) |
| `vae_path` | `null` | Custom VAE path |
| `te_name_or_path` | `null` | Custom TE path |
| `unet_path` | `null` | Custom UNet path |
| `lora_path` | `null` | Apply LoRA during training |
| `assistant_lora_path` | `null` | Accuracy recovery adapter |
| `attn_masking` | `false` | Attention masking (FLUX) |
| `ignore_if_contains` | `null` | Skip layers with strings |
| `only_if_contains` | `null` | Only train layers with strings |
| `compile` | `false` | Use torch.compile |
| `split_model_over_gpus` | `false` | Multi-GPU split |

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  arch: "flux"
  quantize: true
  qtype: "qfloat8"
  quantize_te: true
  qtype_te: "qfloat8"
  layer_offloading: true
  layer_offloading_transformer_percent: 0.8
```

---

## Dataset Config (Hidden Settings)

### Captions & Content

| Setting | Default | Description |
|---------|---------|-------------|
| `default_caption` | `null` | Default if none found |
| `trigger_word` | `null` | Dataset trigger word |
| `random_triggers` | `[]` | Random trigger list |
| `random_triggers_max` | `1` | Max random triggers |
| `caption_ext` | `.txt` | Caption extension |
| `caption_dropout_rate` | `0.0` | Caption dropout |
| `token_dropout_rate` | `0.0` | Per-token dropout |
| `shuffle_tokens` | `false` | Shuffle tokens |
| `keep_tokens` | `0` | First tokens to keep |
| `use_short_captions` | `false` | Use caption_short |
| `replacements` | `[]` | Text replacements |

```yaml
datasets:
  - folder_path: "/path/to/images"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    shuffle_tokens: true
    keep_tokens: 1
```

---

### Image Processing

| Setting | Default | Description |
|---------|---------|-------------|
| `resolution` | `512` | Target resolution(s) |
| `scale` | `1.0` | Image scale |
| `buckets` | `true` | Aspect ratio buckets |
| `bucket_tolerance` | `64` | Bucket tolerance |
| `random_scale` | `false` | Random scale aug |
| `random_crop` | `false` | Random crop |
| `square_crop` | `false` | Force square |
| `flip_x` | `false` | Horizontal flip |
| `flip_y` | `false` | Vertical flip |
| `augmentations` | `null` | Albumentations |

```yaml
datasets:
  - folder_path: "/path/to/images"
    resolution: [512, 768, 1024]
    buckets: true
    flip_x: true
    random_crop: false
```

---

### Masking & Control

| Setting | Default | Description |
|---------|---------|-------------|
| `mask_path` | `null` | Mask images path |
| `alpha_mask` | `false` | Use alpha as mask |
| `invert_mask` | `false` | Invert mask |
| `mask_min_value` | `0.0` | Min mask value |
| `control_path` | `null` | Control images |
| `controls` | `[]` | Auto controls: `depth`, `line`, `pose`, `inpaint`, `mask` |
| `full_size_control_images` | `true` | Don't crop control |

```yaml
datasets:
  - folder_path: "/path/to/images"
    mask_path: "/path/to/masks"
    mask_min_value: 0.1
    invert_mask: false
```

---

### Regularization

| Setting | Default | Description |
|---------|---------|-------------|
| `is_reg` | `false` | Regularization dataset |
| `prior_reg` | `false` | Prior regularization |
| `network_weight` | `1.0` | LoRA weight |
| `loss_multiplier` | `1.0` | Loss multiplier |
| `num_repeats` | `1` | Dataset repeats |

```yaml
datasets:
  - folder_path: "/path/to/reg_images"
    is_reg: true
    network_weight: 0.5
    num_repeats: 1
```

---

### Caching

| Setting | Default | Description |
|---------|---------|-------------|
| `cache_latents` | `false` | Cache in memory |
| `cache_latents_to_disk` | `false` | Cache to disk |
| `cache_clip_vision_to_disk` | `false` | Cache CLIP vision |
| `cache_text_embeddings` | `false` | Cache text embeds |

```yaml
datasets:
  - folder_path: "/path/to/images"
    cache_latents_to_disk: true
    cache_text_embeddings: true
```

---

### Video Training

| Setting | Default | Description |
|---------|---------|-------------|
| `num_frames` | `1` | Frames (>1 = video) |
| `shrink_video_to_frames` | `true` | Even frame sampling |
| `fps` | `24` | FPS for sampling |
| `do_i2v` | `true` | Image-to-video mode |
| `do_audio` | `false` | Load audio |

```yaml
datasets:
  - folder_path: "/path/to/videos"
    num_frames: 16
    shrink_video_to_frames: true
    fps: 24
```

---

### Data Loading

| Setting | Default | Description |
|---------|---------|-------------|
| `num_workers` | `2` | Dataloader workers |
| `prefetch_factor` | `2` | Prefetch factor |
| `poi` | `null` | Point of interest |
| `clip_image_path` | `null` | CLIP reference path |
| `clip_image_from_same_folder` | `false` | CLIP from same folder |
| `extra_values` | `[]` | Extra conditioning |

```yaml
datasets:
  - folder_path: "/path/to/images"
    num_workers: 4
    prefetch_factor: 4
```

---

## Sample Config (Hidden Settings)

| Setting | Default | Description |
|---------|---------|-------------|
| `sampler` | `ddpm` | Sampler type |
| `sample_every` | `100` | Sample interval |
| `width` | `512` | Sample width |
| `height` | `512` | Sample height |
| `neg` | `""` | Negative prompt |
| `seed` | `0` | Starting seed |
| `walk_seed` | `false` | Increment seed |
| `guidance_scale` | `7` | CFG scale |
| `sample_steps` | `20` | Inference steps |
| `network_multiplier` | `1` | LoRA strength |
| `guidance_rescale` | `0.0` | Guidance rescale |
| `num_frames` | `1` | Video frames |
| `fps` | `16` | Video FPS |
| `format` | `jpg` | Output format |
| `do_cfg_norm` | `false` | CFG normalization |

```yaml
sample:
  sampler: "flowmatch"
  sample_every: 250
  width: 1024
  height: 1024
  guidance_scale: 3.5
  sample_steps: 25
  seed: 42
  walk_seed: true
  samples:
    - prompt: "a woman at the beach"
      guidance_scale: 4.0
      sample_steps: 30
    - prompt: "portrait in studio"
      network_multiplier: 0.8
```

---

## Save Config (Hidden Settings)

| Setting | Default | Description |
|---------|---------|-------------|
| `save_every` | `1000` | Save interval |
| `dtype` | `float16` | Save dtype: `float16`, `bf16`, `float32` |
| `max_step_saves_to_keep` | `5` | Max checkpoints |
| `save_format` | `safetensors` | Format: `safetensors`, `diffusers` |
| `push_to_hub` | `false` | Push to HuggingFace |
| `hf_repo_id` | `null` | HF repo ID |
| `hf_private` | `false` | Make HF private |

```yaml
save:
  dtype: "bf16"
  save_every: 500
  max_step_saves_to_keep: 3
  save_format: "diffusers"
  push_to_hub: false
```

---

## EMA Config (Hidden Settings)

| Setting | Default | Description |
|---------|---------|-------------|
| `use_ema` | `false` | Enable EMA |
| `ema_decay` | `0.999` | Decay rate |
| `use_feedback` | `false` | Feed decay back |
| `param_multiplier` | `1.0` | Param multiplier |

```yaml
train:
  ema_config:
    use_ema: true
    ema_decay: 0.99
    use_feedback: false
```

---

## Logging Config

| Setting | Default | Description |
|---------|---------|-------------|
| `log_every` | `100` | Log interval |
| `verbose` | `false` | Verbose logging |
| `use_wandb` | `false` | Enable W&B |
| `use_ui_logger` | `false` | Enable UI logger |
| `project_name` | `ai-toolkit` | W&B project |
| `run_name` | `null` | W&B run name |

```yaml
logging:
  log_every: 10
  use_wandb: true
  project_name: "my-lora-training"
  run_name: "experiment-1"
  use_ui_logger: true
```

---

## Complete Example Config

```yaml
job: extension
config:
  name: "advanced_flux_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "ohwx"
      
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 16
        conv_alpha: 16
        dropout: 0.05
        transformer_only: true
        
      save:
        dtype: bf16
        save_every: 500
        max_step_saves_to_keep: 3
        save_format: diffusers
        
      datasets:
        - folder_path: "/path/to/dataset"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: true
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
          flip_x: true
          num_repeats: 1
          
      train:
        batch_size: 1
        steps: 3000
        gradient_accumulation: 2
        gradient_checkpointing: true
        dtype: bf16
        
        # Prodigy optimizer with warmup-like behavior
        optimizer: "prodigy"
        lr: 1.0
        optimizer_params:
          weight_decay: 0.01
          d_coef: 1.0
          use_bias_correction: true
          safeguard_warmup: true
          growth_rate: 1.02
          
        # LR Scheduler
        lr_scheduler: "constant"
        
        # Timestep control
        timestep_type: "sigmoid"
        content_or_style: "balanced"
        noise_offset: 0.0357
        
        # Loss
        loss_type: "mse"
        min_snr_gamma: 5.0
        
        # EMA
        ema_config:
          use_ema: true
          ema_decay: 0.99
          
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        arch: "flux"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        layer_offloading: true
        layer_offloading_transformer_percent: 0.8
        
      sample:
        sampler: "flowmatch"
        sample_every: 500
        width: 1024
        height: 1024
        guidance_scale: 3.5
        sample_steps: 25
        seed: 42
        walk_seed: true
        samples:
          - prompt: "ohwx person at the beach, sunset"
          - prompt: "portrait of ohwx, studio lighting"
          
      logging:
        log_every: 10
        use_ui_logger: true
```
