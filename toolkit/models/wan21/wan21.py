# WIP, coming soon ish
from functools import partial
import torch
import yaml
import huggingface_hub
from safetensors.torch import load_file
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.memory_management.manager import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import  WanPipeline, WanTransformer3DModel, AutoencoderKL
from .autoencoder_kl_wan import AutoencoderKLWan
import os
import sys

import weakref
import torch
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
import torch
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from typing import TYPE_CHECKING, List
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from toolkit.models.wan21.wan_lora_convert import convert_to_diffusers, convert_to_original
from toolkit.util.quantize import quantize_model
from toolkit.models.loaders.umt5 import get_umt5_encoder

# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.33.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False
}

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False
}


class AggressiveWanUnloadPipeline(WanPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            vae=vae,
            scheduler=scheduler,
        )
        self._exec_device = device
    @property
    def _execution_device(self):
        return self._exec_device
    
    def __call__(
        self: WanPipeline,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None],
                  PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # unload vae and transformer
        vae_device = self.vae.device
        transformer_device = self.transformer.device
        text_encoder_device = self.text_encoder.device
        device = self.transformer.device
        
        print("Unloading vae")
        self.vae.to("cpu")
        self.text_encoder.to(device)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # unload text encoder
        print("Unloading text encoder")
        self.text_encoder.to("cpu")

        self.transformer.to(device)

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(device, transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device, transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(device, transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * \
                        (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # unload transformer
        # load vae
        print("Loading Vae")
        self.vae.to(vae_device)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class Wan21(BaseModel):
    arch = 'wan21'
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = None
    
    _comfy_te_file = ['text_encoders/umt5_xxl_fp16.safetensors', 'text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors']
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(device, model_config, dtype,
                         custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['WanTransformer3DModel']

        # cache for holding noise
        self.effective_noise = None
        
    def get_bucket_divisibility(self):
        return 16

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def load_inference_adapter(self, transformer: WanTransformer3DModel):
        """Load a distilled-step LoRA adapter for faster sampling.
        
        The adapter is kept INACTIVE during training (no effect on training).
        It is only activated during sampling previews by BaseModel.generate_images(),
        allowing fast few-step generation (e.g., 8 steps, CFG 1) with a distilled LoRA.
        
        Supports hybrid adapters that contain both LoRA weights (lora_A/lora_B) and
        direct weight/bias deltas (diff/diff_b) for layers like norms, embeddings, etc.
        """
        lora_path = self.model_config.inference_lora_path
        self.print_and_status_update(f"Loading inference LoRA from {lora_path}")
        
        if not os.path.exists(lora_path):
            # assume it is a hub path like "org/repo/filename.safetensors"
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Inference LoRA path {lora_path} is not a valid local path or hub path. "
                    f"Hub paths should be in the format 'org/repo/filename.safetensors'"
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                self.model_config.inference_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download inference LoRA from {lora_path}: {e}"
                )
        
        lora_state_dict = load_file(lora_path)
        
        # Convert keys to diffusers format if they are in original format
        # ai-toolkit saves Wan 2.1 LoRAs with diffusion_model.* prefix and original
        # attention names (self_attn, q, k, v, etc). convert_to_diffusers() converts
        # these to diffusers format (transformer.*, attn1, to_q, to_k, to_v, etc)
        if any(k.startswith("diffusion_model.") for k in lora_state_dict.keys()):
            lora_state_dict = convert_to_diffusers(lora_state_dict)
        
        # Ensure keys have transformer. prefix for LoRASpecialNetwork
        new_sd = {}
        for key, value in lora_state_dict.items():
            if not key.startswith("transformer."):
                new_key = "transformer." + key
            else:
                new_key = key
            new_sd[new_key] = value
        lora_state_dict = new_sd
        
        # Separate diff weights (direct weight/bias deltas) from LoRA weights.
        # Distilled-step adapters often include diff/diff_b for norms, embeddings,
        # biases, etc. that can't be expressed as LoRA decompositions.
        diff_weights = {}
        lora_weights = {}
        for key, value in lora_state_dict.items():
            if key.endswith('.diff') or key.endswith('.diff_b'):
                diff_weights[key] = value
            else:
                lora_weights[key] = value
        
        if diff_weights:
            self.print_and_status_update(
                f"Found {len(diff_weights)} diff weight keys (norms, biases, embeddings, etc.)"
            )
        
        # Normalize lora_down/lora_up to lora_A/lora_B (peft format)
        # load_weights() expects peft format input when is_transformer=True
        new_sd = {}
        for key, value in lora_weights.items():
            new_key = key.replace('.lora_down.', '.lora_A.')
            new_key = new_key.replace('.lora_up.', '.lora_B.')
            new_sd[new_key] = value
        lora_weights = new_sd
        
        # Auto-detect rank from the state dict
        rank_key = None
        for key in lora_weights.keys():
            if ".lora_A.weight" in key:
                rank_key = key
                break
        if rank_key is None:
            raise ValueError(
                "Could not detect LoRA rank from state dict. "
                "Expected keys containing '.lora_A.weight' or '.lora_down.weight'."
            )
        dim = int(lora_weights[rank_key].shape[0])
        self.print_and_status_update(f"Detected LoRA rank: {dim}")
        
        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": False,
        }
        network_config = NetworkConfig(**network_config)
        
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        
        self.print_and_status_update("Loading inference LoRA weights")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_weights)
        
        # Keep inactive during training — BaseModel.generate_images() will
        # activate this during sampling and deactivate it after
        self.assistant_lora: LoRASpecialNetwork = network
        self.assistant_lora.is_active = False
        # Move to CPU to save VRAM during training
        self.assistant_lora.force_to('cpu', self.torch_dtype)
        
        # Store diff weights for applying during sampling
        # These are direct weight/bias deltas applied to the transformer
        self._inference_diff_weights = {}
        if diff_weights:
            for key, value in diff_weights.items():
                # Convert key: "transformer.blocks.0.attn1.to_k.diff_b" 
                # → module path "blocks.0.attn1.to_k", type "diff_b"
                # Strip "transformer." prefix
                module_key = key
                if module_key.startswith("transformer."):
                    module_key = module_key[len("transformer."):]
                
                self._inference_diff_weights[module_key] = value.to('cpu', dtype=self.torch_dtype)
            
            self.print_and_status_update(
                f"Stored {len(self._inference_diff_weights)} diff weights for sampling"
            )
        
        self.print_and_status_update("Inference LoRA loaded (inactive until sampling)")

    def _apply_diff_weights(self, transformer: WanTransformer3DModel):
        """Apply diff/diff_b weight deltas to the transformer for sampling."""
        if not hasattr(self, '_inference_diff_weights') or not self._inference_diff_weights:
            return
        
        applied = 0
        for key, delta in self._inference_diff_weights.items():
            # key format: "blocks.0.attn1.to_k.diff_b" or "blocks.0.norm3.diff"
            is_bias = key.endswith('.diff_b')
            if is_bias:
                module_path = key[:-len('.diff_b')]
            else:
                module_path = key[:-len('.diff')]
            
            # Navigate to the module
            try:
                module = transformer
                for part in module_path.split('.'):
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
            except (AttributeError, IndexError, TypeError):
                continue
            
            # Apply delta
            delta_dev = delta.to(module.weight.device, dtype=module.weight.dtype)
            if is_bias:
                if module.bias is not None:
                    module.bias.data.add_(delta_dev)
                    applied += 1
            else:
                module.weight.data.add_(delta_dev)
                applied += 1
        
        self.print_and_status_update(f"Applied {applied} diff weights to transformer")

    def _remove_diff_weights(self, transformer: WanTransformer3DModel):
        """Remove diff/diff_b weight deltas from the transformer after sampling."""
        if not hasattr(self, '_inference_diff_weights') or not self._inference_diff_weights:
            return
        
        removed = 0
        for key, delta in self._inference_diff_weights.items():
            is_bias = key.endswith('.diff_b')
            if is_bias:
                module_path = key[:-len('.diff_b')]
            else:
                module_path = key[:-len('.diff')]
            
            try:
                module = transformer
                for part in module_path.split('.'):
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
            except (AttributeError, IndexError, TypeError):
                continue
            
            delta_dev = delta.to(module.weight.device, dtype=module.weight.dtype)
            if is_bias:
                if module.bias is not None:
                    module.bias.data.sub_(delta_dev)
                    removed += 1
            else:
                module.weight.data.sub_(delta_dev)
                removed += 1
        
        self.print_and_status_update(f"Removed {removed} diff weights from transformer")


    def load_wan_transformer(self, transformer_path, subfolder=None):
        self.print_and_status_update("Loading transformer")
        dtype = self.torch_dtype
        transformer = WanTransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.1 models")

        if self.model_config.low_vram:
            # quantize on the device
            transformer.to('cpu', dtype=dtype)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.inference_lora_path is not None:
            self.load_inference_adapter(transformer)
        elif self.model_config.assistant_lora_path is not None:
            raise ValueError(
                "Assistant LoRA (merge for training) is not supported for Wan2.1 models currently. "
                "Use inference_lora_path instead for sampling-only adapters."
            )

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.1 models currently")

        flush()
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent
            )
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to('cpu')

        return transformer

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading Wan model")
        subfolder = 'transformer'
        transformer_path = model_path
        if os.path.exists(transformer_path):
            subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
        
        te_path = "ai-toolkit/umt5_xxl_encoder"   
        if os.path.exists(os.path.join(model_path, 'text_encoder')):
            te_path = model_path
        
        vae_path = self.model_config.extras_name_or_path
        if os.path.exists(os.path.join(model_path, 'vae')):
            vae_path = model_path

        transformer = self.load_wan_transformer(
            transformer_path,
            subfolder=subfolder,
        )

        flush()

        self.print_and_status_update("Loading UMT5EncoderModel")
        
        tokenizer, text_encoder = get_umt5_encoder(
            model_path=te_path,
            tokenizer_subfolder="tokenizer",
            encoder_subfolder="text_encoder",
            torch_dtype=dtype,
            comfy_files=self._comfy_te_file
        )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing UMT5EncoderModel")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_text_encoder_percent > 0:
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent
            )

        if self.model_config.low_vram:
            print("Moving transformer back to GPU")
            # we can move it back to the gpu now
            transformer.to(self.device_torch)

        scheduler = Wan21.get_train_scheduler()
        self.print_and_status_update("Loading VAE")
        # todo, example does float 32? check if quality suffers
        
        if self._wan_vae_path is not None:
            # load the vae from individual repo
            vae = AutoencoderKLWan.from_pretrained(
                self._wan_vae_path, torch_dtype=dtype).to(dtype=dtype)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                vae_path, subfolder="vae", torch_dtype=dtype).to(dtype=dtype)
        flush()

        self.print_and_status_update("Making pipe")
        pipe: WanPipeline = WanPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        self.pipeline = pipe
        self.model = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        if self.model_config.low_vram:
            pipeline = AggressiveWanUnloadPipeline(
                vae=self.vae,
                transformer=self.model,
                transformer_2=self.model,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                expand_timesteps=self._wan_expand_timesteps,
                device=self.device_torch
            )
        else:
            pipeline = WanPipeline(
                vae=self.vae,
                transformer=self.unet,
                transformer_2=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                expand_timesteps=self._wan_expand_timesteps,
                scheduler=scheduler,
            )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: WanPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # vae_scale_factor_spatial = 8
        # vae_scale_factor_temporal = 4
        # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # shape = (
        #     batch_size,
        #     num_channels_latents, # 16
        #     num_latent_frames,  # 81
        #     int(height) // self.vae_scale_factor_spatial,
        #     int(width) // self.vae_scale_factor_spatial,
        # )

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        return PromptEmbeds(prompt_embeds)

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device('cpu'):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]

        # Normalize shapes
        norm_images = []
        for image in image_list:
            if image.ndim == 3:
                # (C, H, W) -> (C, 1, H, W)
                norm_images.append(image.unsqueeze(1))
            elif image.ndim == 4:
                # (T, C, H, W) -> (C, T, H, W)
                norm_images.append(image.permute(1, 0, 2, 3))
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        # Stack to (B, C, T, H, W)
        images = torch.stack(norm_images)
        B, C, T, H, W = images.shape

        # Resize if needed (B * T, C, H, W)
        if H % 8 != 0 or W % 8 != 0:
            target_h = H // 8 * 8
            target_w = W // 8 * 8
            images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images = images.view(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std

        return latents.to(device, dtype=dtype)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: Wan21 = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        return convert_to_original(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return convert_to_diffusers(state_dict)
    
    def get_base_model_version(self):
        return "wan_2.1"
    
    def get_transformer_block_names(self):
        return ['blocks']
