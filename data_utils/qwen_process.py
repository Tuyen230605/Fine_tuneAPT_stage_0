import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from typing import List, Optional
from torch import Tensor

def data2qwenmsg(lang: str, num_cam: int):
    img_msg = {
                "type": "image",
                "image": None,
                }
    text_msg = {"type": "text", "text": f""}
    messages = [
        {
            "role": "user",
            "content": [
                img_msg,
                text_msg
            ],
        },
        # {"role": "assistant", "content": f''},
    ]

    messages[0]['content'][-1]['text'] = lang
    if num_cam > 1:
        for _ in range(1, num_cam):
            messages[0]['content'].insert(0, img_msg)

    return messages

def process_batch_images(obs_rgbs: Tensor):
    B, To, ncam, C, H, W = obs_rgbs.shape
    obs_rgbs = obs_rgbs.reshape(B, To*ncam, C, H, W)
    # for each batch size, transform obs_rgbs to PIL Image list
    obs_rgbs_list = []
    for i in range(B):
        image_list = [ToPILImage()(obs_rgbs[i, j]) for j in range(To*ncam)]
        obs_rgbs_list.append(image_list)
    return obs_rgbs_list



def process_batch_to_qwen(tokenizer, 
                          multimodal_processor, 
                          obs_rgbs: Tensor, 
                          prompt_texts: List[str], 
                          max_input_ids_len: int=512,
                          obs_masks: Optional[Tensor] = None):
    
    # obs_rgbs: (B, To, ncam, 3, H, W)
    B, To, ncam, C, H, W = obs_rgbs.shape
    obs_rgbs = obs_rgbs.reshape(B*To, ncam, C, H, W)
    # if obs_masks is not None:
    #     obs_masks = obs_masks.reshape(B*To, ncam, H, W)

    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    image_grid_thw_list = []
    for i, prompt_text in enumerate(prompt_texts):
        # if obs_masks is not None:
        #     obs_mask = obs_masks[i] # (ncam, H, W)
        #     cam_valid = obs_mask.sum(dim=-1).sum(dim=-1).nonzero()
        #     ncam_valid = cam_valid.shape[0]
        # else:
        #     ncam_valid = ncam
        
        message = data2qwenmsg(prompt_text, ncam)
        text = multimodal_processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        # image_data = obs_rgbs[i*To:(i+1)*To, cam_valid.squeeze()] # [To, ncam_valid, C, H, W]
        image_data = obs_rgbs[i*To:(i+1)*To] # [To, ncam, C, H, W]
        image_data = image_data.reshape(-1, C, H, W)
        image_list = []
        for each in image_data:
            each = ToPILImage()(each)
            image_list.append(each)
        
        model_inputs = multimodal_processor(
            text=[text],
            images=image_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        padding_input_ids = torch.nn.functional.pad(
            input=model_inputs["input_ids"],
            pad=(max_input_ids_len - model_inputs["input_ids"].shape[1], 0), 
            mode='constant',
            value=tokenizer.pad_token_id
        )
        attention_mask = padding_input_ids.ne(tokenizer.pad_token_id)
        
        # print(padding_input_ids.shape)
        # print(attention_mask.shape)
        input_ids_list.append(padding_input_ids.squeeze(0))
        attention_mask_list.append(attention_mask.squeeze(0))
        pixel_values_list.append(model_inputs["pixel_values"])
        image_grid_thw_list.append(model_inputs["image_grid_thw"])
    
    input_ids = torch.stack(input_ids_list, dim=0).to(obs_rgbs.device) # (B, L)
    attention_masks = torch.stack(attention_mask_list, dim=0).to(obs_rgbs.device) # (B, L)
    pixel_values = torch.stack(pixel_values_list, dim=0).to(obs_rgbs.device) # (B, 768, 1536)
    image_grid_thws = torch.stack(image_grid_thw_list, dim=0).to(obs_rgbs.device) # (B, 3, 3)

    return input_ids, attention_masks, pixel_values, image_grid_thws