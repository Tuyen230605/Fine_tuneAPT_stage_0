import time
import torch
from einops import rearrange
from torch import nn, Tensor
from functools import partial
from typing import List, Optional, Dict
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision.transforms import v2, InterpolationMode
from transformers.feature_extraction_utils import BatchFeature
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2VLImageProcessorFast, Qwen2TokenizerFast
from .modality import ModalityType


VALID_VLM_FINETUNE_MODES = ("frozen", "lora", "full")


class ExtentedEmbedding(nn.Module):
    def __init__(
        self, 
        old_embedding: nn.Embedding, 
        old_vocab_size: int, 
        additional_vocab_size: int
    ):
        super().__init__()
        self.old_vocab_size = old_vocab_size
        self.old_embedding = old_embedding
        self.new_embedding = nn.Embedding(
            additional_vocab_size,
            self.old_embedding.embedding_dim,
            max_norm=self.old_embedding.max_norm,
            norm_type=self.old_embedding.norm_type,
            scale_grad_by_freq=self.old_embedding.scale_grad_by_freq,
            sparse=self.old_embedding.sparse,
        ).to(
            device=self.old_embedding.weight.device,
            dtype=self.old_embedding.weight.dtype,
        )
        with torch.no_grad():
            mean_embedding = self.old_embedding.weight.mean(dim=0, keepdim=True)
            self.new_embedding.weight.copy_(mean_embedding.expand_as(self.new_embedding.weight))
    
    def forward(self, input_ids: Tensor):
        clipped_input_ids = input_ids.clip(None, self.old_vocab_size-1)
        outputs = self.old_embedding(clipped_input_ids).contiguous()

        mask = input_ids >= self.old_vocab_size
        if mask.any():
            masked_outputs = self.new_embedding(input_ids[mask] - self.old_vocab_size)
            if masked_outputs.dtype != outputs.dtype:
                outputs = outputs.type_as(masked_outputs)
            outputs[mask] = masked_outputs
        return outputs


class QwenVL3Extractor(nn.Module):
    def __init__(
        self,
        add_action_id: bool,
        vlm_finetune_mode: str = "frozen",
        use_gradient_checkpointing: bool = False,
        fp16: bool = True,
    ):
        super().__init__()
        assert vlm_finetune_mode in VALID_VLM_FINETUNE_MODES, \
            f"vlm_finetune_mode must be one of {VALID_VLM_FINETUNE_MODES}, got {vlm_finetune_mode!r}"
        
        lora_config = LoraConfig(
            r=16,  
            lora_alpha=16,  
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"], 
            bias="none", 
            task_type="CAUSAL_LM",
        )

        # Prefer flash_attention_2 for speed; fall back to sdpa if unavailable
        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass
        
        self.fp16 = fp16
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", 
            attn_implementation=attn_impl, 
            torch_dtype=torch.bfloat16 if self.fp16 else torch.float32
        )
        self.config = qwen.config
        
        self.processor = Qwen3VLProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            # cache_dir="~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct",
            # local_files_only=True,
        )
        
        # add special token: <action>
        self.add_action_id = add_action_id
        self.tokenizer: Qwen2TokenizerFast = self.processor.tokenizer

        if self.add_action_id:
            qwen.model.language_model.embed_tokens = ExtentedEmbedding(
                old_embedding=qwen.model.language_model.embed_tokens,
                old_vocab_size=max(self.tokenizer.get_vocab().values()) + 1,
                additional_vocab_size=1
            )
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<action>"]})

        # disable image value scaling
        self.image_processor: Qwen2VLImageProcessorFast = self.processor.image_processor
        # the original image processor does rescale with scaling factor 1/255.
        # since the images here are already normalized to 0~1, we don't need rescale anymore
        self.image_processor.do_rescale = False
        self.image_processor.rescale_factor = 1.0

        # forward hook of each llm layer
        self.hidden_states_buffer = [None for _ in qwen.language_model.layers]
        def llm_per_layer_forward_hook(layer_idx, module, input, output):
            self.hidden_states_buffer[layer_idx] = output
        for layer_idx, module in enumerate(qwen.language_model.layers):
            module.register_forward_hook(partial(llm_per_layer_forward_hook, layer_idx))
        
        # forward hook of the whole llm model
        self.llm_inputs = None
        def llm_whole_forward_hook(module, args, kwargs, output):
            self.llm_inputs = kwargs
            # kwargs are:
            # 'input_ids',
            # 'position_ids',
            # 'attention_mask',
            # 'past_key_values',
            # 'inputs_embeds',
            # 'cache_position',
            # 'visual_pos_masks',
            # 'deepstack_visual_embeds'
        qwen.language_model.register_forward_hook(
            llm_whole_forward_hook, with_kwargs=True)

        # configure trainable parameters based on finetune mode
        self.vlm_finetune_mode = vlm_finetune_mode
        self._qwen = qwen
        self.qwen_peft: Optional[PeftModel] = None
        if vlm_finetune_mode == "lora":
            self._qwen.requires_grad_(False)
            self.qwen_peft = get_peft_model(self._qwen, lora_config)
            self._set_extra_token_trainability(self.add_action_id)
            print("[INFO] In Qwen3VL (LoRA mode):")
            self.qwen_peft.print_trainable_parameters()
            self._log_trainable_parameter_summary("LoRA")
        elif vlm_finetune_mode == "full":
            self._qwen.requires_grad_(True)
            self._cast_sensitive_params_to_fp32()
            self._set_extra_token_trainability(self.add_action_id)
            self._log_trainable_parameter_summary("Full finetune")
        else:  # frozen
            self._qwen.requires_grad_(False)
            self._set_extra_token_trainability(False)
            self._qwen.eval()
            self._log_trainable_parameter_summary("Frozen")

        # Gradient checkpointing trades ~30% speed for ~60% activation memory savings.
        # Only meaningful when VLM has trainable parameters.
        if use_gradient_checkpointing and vlm_finetune_mode != "frozen":
            target = self.qwen_peft if self.qwen_peft is not None else self._qwen
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f"[INFO] Gradient checkpointing enabled for Qwen3VL ({vlm_finetune_mode} mode)")

        print(f"[INFO] Qwen3VL using attention: {attn_impl}")
    
        self.layer_times = {}

    @property
    def enable_lora(self) -> bool:
        return self.vlm_finetune_mode == "lora"

    @property
    def qwen(self) -> Qwen3VLForConditionalGeneration:
        return self._qwen

    # Parameters kept in fp32 during full fine-tuning, following openpi convention.
    # Everything else stays in bf16 (as loaded by from_pretrained).
    #
    # Why these groups:
    #   norm       — RMSNorm/LayerNorm scale weights; small updates (≈1e-6/step) are
    #                lost in bf16 rounding; fp32 storage lets the optimizer accumulate
    #                them without stochastic quantisation.
    #   patch_embed — Conv3d projecting raw pixels → visual tokens; analogous to
    #                 word embeddings (each patch position is a "lookup"), benefits from
    #                 fp32 for the same reasons.
    #   pos_embed  — Learned 2-D position biases for the vision encoder.
    #   embed_tokens — LLM token embedding table; sparse updates (only seen tokens get
    #                  gradients) make fp32 important to preserve unseen-token embeddings.
    #
    # Forward-pass dtype: the surrounding torch.autocast("cuda", bfloat16) context
    # will cast fp32 outputs back to bf16 at the next linear/matmul op, so these
    # fp32 params do NOT cascade the whole network to fp32.
    #
    # DeepSpeed note: BF16_Optimizer copies fp32 master → model param via .data.copy_(),
    # which preserves the target tensor's dtype.  fp32 params therefore receive full-
    # precision updates; bf16 params still get stochastic-rounded updates as before.
    _FP32_KEYWORDS = ("norm", "patch_embed", "pos_embed", "embed_tokens")

    def _cast_sensitive_params_to_fp32(self):
        n_cast = 0
        for name, param in self._qwen.named_parameters():
            if any(kw in name.lower() for kw in self._FP32_KEYWORDS):
                param.data = param.data.float()
                n_cast += 1
        print(f"[INFO] Qwen3VL (full mode): {n_cast} sensitive params cast to fp32 "
              f"(norm / patch_embed / pos_embed / embed_tokens); "
              f"remaining params stay bf16.")

    def _set_extra_token_trainability(self, enabled: bool):
        embed_tokens = self._qwen.model.language_model.embed_tokens
        if isinstance(embed_tokens, ExtentedEmbedding):
            embed_tokens.new_embedding.weight.requires_grad_(enabled)

    def _log_trainable_parameter_summary(self, mode_name: str):
        total_params = sum(p.numel() for p in self._qwen.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "[INFO] In Qwen3VL ({} mode): {:.1f}M / {:.1f}M parameters trainable".format(
                mode_name,
                trainable_params / 1e6,
                total_params / 1e6,
            )
        )
    
    @property
    def image_ds(self):
        config = self.config.vision_config
        return config.patch_size * config.spatial_merge_size
    
    @property
    def num_llm_layers(self) -> int:
        return self.config.text_config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        return self.config.text_config.hidden_size

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vlm_finetune_mode == "frozen":
            self._qwen.eval()
        return self

    def make_conversation_for_one_sample(
        self, 
        images: Tensor, 
        camera_masks: Optional[Tensor], 
        prompt: str
    ):
        Ncam, C, H, W = images.shape
        # message = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "./images/droid0.png"},
        #             {"type": "image", "image": "./images/droid1.png"},
        #             {"type": "text", "text": "Given the two images from the same timestep but different viewpoint, what is the robot doing?"},
        #         ],
        #     }
        # ]
 
        content = []
        for n in range(Ncam):
            if camera_masks is None or camera_masks[n]:
                content.append({"type": "image", "image": images[n]})
        content.append({"type": "text", "text": prompt})

        message = [
            {
                "role": "user",
                "content": content
            }
        ]
        return message
    
    @torch.no_grad()
    def chat(
        self, 
        images: Tensor, 
        camera_masks: Optional[Tensor], 
        prompts: List[str], 
        generation_config: dict,
        add_action_id: bool = False,
    ):
        """
        Args:
            mv_images (Tensor): (B, Ncam, C, H, W),
            prompts (List[str]): prompt text
        """
        if camera_masks is None:
            B, Ncam, _, _, _ = images.shape
            camera_masks = torch.ones((B, Ncam), dtype=torch.bool)
        
        conversations = [self.make_conversation_for_one_sample(I, M, L) 
                         for I, M, L in zip(images, camera_masks, prompts)]
        inputs: BatchFeature = self.processor.apply_chat_template(
            conversation=conversations,
            tokenize=True,
            add_generation_prompt=True, # if True, will add ["<|im_start|>", "assistant", 'Ċ' (id=198)]
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        )
        inputs = inputs.to(images.device)

        # add special <action> token
        if add_action_id and self.add_action_id:
            custom_ends = ["<action>"]
            B = inputs["input_ids"].shape[0]
            custom_ids: Tensor = inputs["input_ids"].new_zeros((B, len(custom_ends)))
            for i in range(len(custom_ends)):
                if isinstance(custom_ends[i], int):
                    custom_ids[:, i] = custom_ends[i]
                else:
                    custom_ids[:, i] = self.tokenizer.convert_tokens_to_ids(custom_ends[i])
            inputs["input_ids"] = torch.cat([inputs["input_ids"], custom_ids], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], inputs["attention_mask"].new_ones((B, len(custom_ends)))], dim=-1)
            assert inputs["input_ids"].shape == inputs["attention_mask"].shape

        with torch.autocast("cuda", torch.bfloat16):
            qwen = self.qwen_peft if self.enable_lora else self._qwen
            generated_ids = qwen.generate(**inputs, **generation_config)
        questions = [out_ids[:len(in_ids)] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        responses = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        questions = self.processor.batch_decode(questions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        responses = self.processor.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # questions = self.processor.batch_decode(questions, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # responses = self.processor.batch_decode(responses, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        return questions, responses

    def hidden_layer_features(
        self, 
        images: Tensor, 
        camera_masks: Optional[Tensor], 
        prompts: List[str],
        select_layer_indices: List[int],
        fp16: bool
    ) -> Dict[str, Tensor]:
        """
        Args:
            mv_images (Tensor): (B, Ncam, C, H, W),
            prompts (List[str]): prompt text
        """
        if camera_masks is None:
            B, Ncam, _, _, _ = images.shape
            camera_masks = torch.ones((B, Ncam), dtype=torch.bool)
        
        conversations = [self.make_conversation_for_one_sample(I, M, L) 
                         for I, M, L in zip(images, camera_masks, prompts)]
        inputs = self.processor.apply_chat_template(
            conversation=conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        )
        inputs = inputs.to(images.device)
        
        # add special <action> token
        if self.add_action_id:
            custom_ends = ["<action>"]
            B = inputs["input_ids"].shape[0]
            custom_ids: Tensor = inputs["input_ids"].new_zeros((B, len(custom_ends)))
            for i in range(len(custom_ends)):
                if isinstance(custom_ends[i], int):
                    custom_ids[:, i] = custom_ends[i]
                else:
                    custom_ids[:, i] = self.tokenizer.convert_tokens_to_ids(custom_ends[i])
            inputs["input_ids"] = torch.cat([inputs["input_ids"], custom_ids], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], inputs["attention_mask"].new_ones((B, len(custom_ends)))], dim=-1)
            assert inputs["input_ids"].shape == inputs["attention_mask"].shape

        # use only the first `n = use_num_layers` llm layers to extract feature
        max_layer_num = max(select_layer_indices) + 1
        qwen: Qwen3VLForConditionalGeneration = self.qwen
        assert max_layer_num <= len(qwen.language_model.layers)
        original_llm_layers = qwen.language_model.layers
        qwen.language_model.layers = original_llm_layers[:max_layer_num]
        
        with torch.autocast("cuda", torch.bfloat16 if fp16 else torch.float32):
            if self.enable_lora:
                with self.qwen_peft._enable_peft_forward_hooks(**inputs):
                    kwargs = {k: v for k, v in inputs.items() if k not in self.qwen_peft.special_peft_forward_args}
                    qwen.model(**kwargs)
            else:
                qwen.model(**inputs)

        qwen.language_model.layers = original_llm_layers

        hidden_states = [self.hidden_states_buffer[i] for i in select_layer_indices]
        # list len = n_layers, each contains a tensor with shape (B, L, C)

        input_ids = inputs.input_ids  # (B, L, C)
        attention_mask: Tensor = inputs.attention_mask  # (B, L)
        # is_vision_mask = input_ids == self.qwen.config.image_token_id  # (B, L)
        is_vision_mask = self.llm_inputs["visual_pos_masks"]  # (B, L)

        L = input_ids.shape[1]
        offset = torch.cumsum(is_vision_mask.int(), dim=-1)  # (B, L)
        pos_id_1d = torch.arange(L).to(offset) - (offset - 1).clamp_min(0)
        # P P P V V V V L L L L L (P for <pad>, V for <vision>, L for <language>)
        # 0 1 2 3 3 3 3 4 5 6 7 8

        return {
            "deepstack_visual_embeds": self.llm_inputs["deepstack_visual_embeds"],
            "inputs_embeds": self.llm_inputs["inputs_embeds"], 
            "hidden_states": hidden_states,
            "attention_mask": attention_mask.bool(),
            "is_vision_mask": is_vision_mask,
            "pos_id_1d": pos_id_1d,
        }
    
    def retreive_input_vision_features(
        self,
        images: Tensor, 
        camera_masks: Tensor, 
        input_embeds: Tensor,
        is_vision_mask: Tensor, 
    ):
        """
        Args:
            image: (B, N, C, H, W)
            camera_masks: (B, N)
            input_embeds: (B, L, C)
            is_vision_mask: (B, L)
        """
        B, N, _, H, W = images.shape
        Hds, Wds = H // self.image_ds, W // self.image_ds
        
        C = input_embeds.shape[-1]
        placeholder = input_embeds.new_zeros(B, N, Hds*Wds, C)
        placeholder[camera_masks] = input_embeds[is_vision_mask].view(-1, Hds*Wds, C)
        return placeholder
    
    def retreive_deepstack_visual_features(
        self,
        images: Tensor, 
        camera_masks: Tensor,
        deepstack_visual_embeds: List[Tensor]
    ):
        """
        Args:
            image: (B, N, C, H, W)
            camera_masks: (B, N)
            deepstack_visual_embeds: List of (valid_BN*dsH*dsW, C)
        """
        B, N, _, H, W = images.shape
        Hds, Wds = H // self.image_ds, W // self.image_ds
        
        placeholders = []
        for v in deepstack_visual_embeds:
            C = v.shape[-1]
            placeholder = v.new_zeros(B, N, Hds*Wds, C)
            placeholder[camera_masks] = v.view(-1, Hds*Wds, C)
            placeholders.append(placeholder)
        return placeholders

    def retreive_latent_planning_token(self, hidden_states: Tensor):
        """
        Args:
            hidden_states: (B, L, C)
        """
        if self.add_action_id:
            return hidden_states[:, -2:].flatten(start_dim=1)  # (B, 2*C)
        else:
            return hidden_states[:, -1]  # (B, C)


class QwenVL3Encoder(QwenVL3Extractor):
    def __init__(
        self,
        add_action_id: bool,
        vlm_finetune_mode: str = "frozen",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(add_action_id, vlm_finetune_mode, use_gradient_checkpointing)
        self.pool_aux = nn.AvgPool2d(self.image_ds)
        self.resize0 = v2.Resize((256, 256), InterpolationMode.NEAREST)
        self.resize2 = v2.Resize((256, 256), InterpolationMode.BILINEAR)
        self.resize3 = v2.Resize((256, 256), InterpolationMode.BICUBIC)
    
    def encode_aux(self, a: Tensor):
        B, C, H, W = a.shape
        assert H == W
        
        bool_type = a.dtype == torch.bool
        float_type = a.is_floating_point()
        int_type = (not bool_type) and (not float_type)
        
        if H != 256 or W != 256:
            if float_type:
                a_resize = self.resize2(a)
            else:
                a_resize = self.resize0(a.float())
        else:
            a_resize = a if float_type else a.float()
        
        a_ds: Tensor = self.pool_aux(a_resize)
        if bool_type:
            a_ds = a_ds > 1e-4
        elif int_type:
            a_ds = a_ds.to(a.dtype)
        a_ds = rearrange(a_ds, "b c h w -> b (h w) c")
        return a_ds
    
    def pool_mv_masks(self, mask: Tensor):
        if mask is not None:
            B, T, N, H, W = mask.shape
            mask = rearrange(mask, "b t n h w -> (b t n) () h w")
            mask_ds = self.encode_aux(mask)
            mask_ds = rearrange(mask_ds, "(b t n) l 1 -> b t n l", b=B, t=T, n=N)
        else:
            mask_ds = None
        return mask_ds
    
    def pool_mv_aux(self, aux: Tensor):
        if aux is not None:
            B, T, N, C, H, W = aux.shape
            aux = rearrange(aux, "b t n c h w -> (b t n) c h w")
            aux_ds = self.encode_aux(aux)
            aux_ds = rearrange(aux_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        else:
            aux_ds = None
        return aux_ds

    def forward(
        self, 
        rgb: Tensor, 
        mask: Tensor, 
        norm_xy: Tensor, 
        text: List[str],
        select_layer_indices: List[int],
        fp16: bool,
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            mask (Tensor): (B, T, N, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            text (List[Tensor]): (B,)

        Returns:
            inputs (Dict[str, Tensor]): inputs
            outputs (Dict[str, Tensor | List[Tensor]]):
                - camera_mask (Tensor): (B, Ncam)
                - norm_xy_ds (Tensor): (B, Ncam, Lv, 2)
                - valid_vision_mask (Tensor): (B, Ncam, Lv)
                - deepstack_visual_embeds (List[Tensor]): len = n_stack_layers = 3, shape = (S, C), S = sum(camera_mask) * Hds * Wds
                - inputs_embeds (Tensor): shape = (B, L, C), L = Lpad + Ncam*Lv + Llang
                - hidden_states (List[Tensor]): len = n_layers, shape = (B, L, C), L = Lpad + Ncam*Lv + Llang
                - attention_mask (Tensor): (B, L)
                - is_vision_mask (Tensor): (B, L)
                - pos_id_1d (Tensor): (B, L)
                - modality_type: (B, L), including PAD, VISION, LANGUAGE
                # - visual_embeds (List[Tensor]): len = 3 (n_deep_stack) + 1 (last_layer), shape = (B, Ncam, Lv, C)
                # - latent_planning_token (Tensor): (B, 2*C)
        """
        assert rgb.ndim == 6
        if mask is not None:
            assert mask.ndim == 5
        assert norm_xy.ndim == 6

        if mask is not None:
            mask_ds = self.pool_mv_masks(mask[:, -1:])  # (B, T=1, N, L)
            camera_mask = mask_ds.any(dim=-1)  # (B, T=1, N)
        else:
            mask_ds = None
            camera_mask = None
        
        norm_xy_ds = self.pool_mv_aux(norm_xy[:, -1:])  # (B, T=1, N, L, 2)

        outputs = self.hidden_layer_features(
            images=rgb[:, -1],  # (B, N, C, H, W)
            camera_masks=camera_mask[:, -1],  # (B, N)
            prompts=text,
            select_layer_indices=select_layer_indices,
            fp16=fp16
        )
        
        inputs = {
            "rgb": rgb,
            "mask": mask,
            "norm_xy": norm_xy,
            "text": text,
        }
        
        outputs.update({
            "norm_xy_ds": norm_xy_ds[:, -1],      # (B, Ncam, Lv, 2) 
            "camera_mask": camera_mask[:, -1],    # (B, Ncam)
            "valid_vision_mask": mask_ds[:, -1],  # (B, Ncam, Lv)
        })

        # 对于token进行更细粒度的掩码，支持在每一幅图像中都注设置局部掩码
        outputs["attention_mask"][outputs["is_vision_mask"]] &= outputs["valid_vision_mask"][outputs["camera_mask"]].ravel()

        # 记录每个token的modality代表什么
        attention_mask = outputs["attention_mask"]  # bool Tensor
        is_vision_mask = outputs["is_vision_mask"]  # bool Tensor
        modality_type = torch.full(attention_mask.shape, ModalityType.LANGUAGE, 
                                   dtype=torch.long, device=attention_mask.device)  # language as default
        modality_type[is_vision_mask] = ModalityType.VISION     # vision
        modality_type[~attention_mask] = ModalityType.PAD       # padding
        outputs["modality_type"] = modality_type

        return inputs, outputs


# prefix = ("Suppose you are the robot or the human in the images, "
#                   "what should you do next to complete the task: ")


if __name__ == "__main__":

    import cv2
    import numpy as np
    from transformers import Qwen2VLImageProcessorFast

    def read_image(path: str):
        bgr = cv2.imread(path)
        bgr = cv2.resize(bgr, (256, 256))
        rgb = np.ascontiguousarray(bgr[:, :, [2, 1, 0]])
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # (C, H, W)
    

    # images = [[read_image("./images/demo0.png"), torch.zeros(3, 256, 256)],
    #           [read_image("./images/droid0.png"), read_image("./images/droid1.png")]]
    images = [[read_image("./images/pick_place_scene2.png"), torch.zeros(3, 256, 256)],
              [read_image("./images/droid0.png"), read_image("./images/droid1.png")]]
    images = torch.stack([torch.stack(r, dim=0) for r in images], dim=0).cuda()  # (B, N, C, H, W)
    camera_masks = torch.tensor([[1, 0],
                                 [1, 1]]).bool().cuda()
    norm_xy = torch.randn(2, 2, 2, 256, 256).cuda()
    image_masks = torch.ones(2, 2, 256, 256).bool().cuda()
    image_masks[0, 0, -33:, -33:] = False
    image_masks[~camera_masks] = False
    # prompts = ["Describe this image in short.",
    #            "Given the two images from the same timestep but different viewpoints, describe what is the robot doing?"]
    
    prompts = ["Your task is to: <pick up the eggplant and place it on the plate>. Locate the key object needed and then figure out how to execute the task.",
               "Your task is to: <clean up the kitchen>. Locate the key object needed and then figure out how to execute the task."]
    
    print(images.shape, images.dtype)

    qwen = QwenVL3Encoder(add_action_id=False, vlm_finetune_mode="frozen").cuda()
    gen_config = {"max_new_tokens": 256}
    # gen_config = {"max_new_tokens": 256, "num_beams": 1, "do_sample": False}
    questions, responses = qwen.chat(images, camera_masks, prompts, gen_config)

    inputs, outputs = qwen(
        rgb=images.unsqueeze(1),
        mask=image_masks.unsqueeze(1),
        norm_xy=norm_xy.unsqueeze(1),
        text=prompts,
        select_layer_indices=list(range(20))
    )

    for q, r in zip(questions, responses):
        print("-"*61)
        print(q)
        print(r)
    
    trainable_params = [p for p in qwen.named_parameters() if p[1].requires_grad]
    size = sum([x[1].numel() for x in trainable_params])

    from IPython import embed
    embed()


