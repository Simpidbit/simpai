import torch
import torch.nn.functional as F

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from simpai import logger, visual

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

def chw_to_qwen_pixel_values_match_processor(img_chw: torch.Tensor, patch_size: int, grid_hw: int = 14):
    """
    输出对齐 processor: pixel_values [196,1176], image_grid_thw [[1,14,14]]
    假设 1176 = 2 * (3*patch_size*patch_size)
    """
    device = img_chw.device
    dtype = img_chw.dtype

    # 目标是产生 2*grid_hw*grid_hw 个“基础token”，然后两两拼接
    base_tokens = 2 * grid_hw * grid_hw  # 392
    base_dim = 3 * patch_size * patch_size  # 588 (若 patch_size=14)
    out_dim = 2 * base_dim  # 1176

    # 我们把图 resize 成能切出 base_tokens 个 patch 的形状：
    # 最简单：把高度方向翻倍，相当于 (2*grid_hw, grid_hw) 网格
    target_h = (2 * grid_hw) * patch_size
    target_w = grid_hw * patch_size

    x = img_chw.unsqueeze(0)  # [1,3,H,W]
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    x = x.squeeze(0)  # [3, target_h, target_w]

    gh = 2 * grid_hw
    gw = grid_hw

    # patchify -> [gh*gw, base_dim] = [392,588]
    x = x.view(3, gh, patch_size, gw, patch_size).permute(1,3,0,2,4).contiguous()
    pv = x.view(gh * gw, base_dim)  # [392,588]

    # pack 两两拼接 -> [196,1176]
    pv = pv.view(grid_hw * grid_hw, 2, base_dim).reshape(grid_hw * grid_hw, out_dim)

    image_grid_thw = torch.tensor([[1, grid_hw, grid_hw]], device=device, dtype=torch.long)
    return pv, image_grid_thw

def qwen_score(img: torch.Tensor, processor, text_prompt:str, device, qwen_model) -> torch.Tensor:
    # 可微地构造视觉输入
    vc = qwen_model.config.vision_config
    patch_size = vc.patch_size
    merge_size = getattr(vc, "spatial_merge_size", getattr(vc, "merge_size", 1))
    
    grid_hw = 14
    num_image_tokens = (grid_hw * grid_hw) // (merge_size**2)
    full_prompt = ("<|image_pad|> " * num_image_tokens) + text_prompt
    
    tokenizer = processor.tokenizer
    text_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    pixel_values, image_grid_thw = chw_to_qwen_pixel_values_match_processor(
        img.to(device), patch_size=patch_size, grid_hw=grid_hw
    )

    '''
    tmp = processor(text=[text_prompt], images=img.unsqueeze(0), return_tensors="pt", do_rescale=False)
    print("processor pixel_values:", tmp["pixel_values"].shape)
    print("yours pixel_values:", pixel_values.shape)
    print("processor image_grid_thw:", tmp["image_grid_thw"])
    print("yours image_grid_thw:", image_grid_thw)
    '''
    
    out = qwen_model(**text_inputs, pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
    output_logits = out.logits
    digit_logits = output_logits[0, -1, 15:25]

    probs = F.softmax(digit_logits, dim = -1)
    weights = torch.arange(10, device = probs.device, dtype = probs.dtype)
    score = torch.tensor(9., device = probs.device, dtype = probs.dtype) - (probs * weights).sum()
    
    return score

class QwenScorer(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self.device: torch.device = device
        self.text_prompt: str = "这是一张低光图像恢复模型的输出图，请用一个整数描述这张图片的质量（范围从0到9，越高质量越好），以指导低光图像恢复模型的训练。要考虑的因素：图像的光照质量、图像的分辨率质量、图像中的物体轮廓是否清晰。注意：只生成一个整数，其他任何token都不要生成！！禁止生成除了整数之外的任何自然语言！！请输出:"

        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype = torch.float32,
            device_map = self.device,          # 也可以用 "auto"（多卡/大模型时更方便）
            low_cpu_mem_usage = True,           # 这个参数设为True有什么用？
        ).eval()
        for p in self.qwen_model.parameters():
            p.requires_grad_(False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 3:
            img = torch.unsqueeze(img, dim = 0)

        img_cnt = img.shape[0]

        aver_score: torch.Tensor = torch.tensor(0., dtype = torch.float32, device = self.device)
        for i in range(img_cnt):
            #visual.show_chw(img[i])
            score = qwen_score(img[i], self.processor, self.text_prompt, self.device, self.qwen_model)
            logger.debug(f'Qwen score: {score} for img[{i}]')
            aver_score += score
        aver_score /= img_cnt

        return aver_score
