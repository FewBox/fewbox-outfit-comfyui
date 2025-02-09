import torch
import numpy as np
from PIL import Image
from .captions import CAPTION

def tensor_to_pil(tensor, is_mask=False):
    print("ndim", tensor.ndim)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if is_mask:
        print("shape", tensor.shape)
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        # 确保张量是单通道的
        if tensor.ndim != 2:
            raise ValueError("Mask tensor must have shape [H, W] or [1, H, W]")
        # 将张量的值范围从 [0, 1] 转换为 [0, 255]
        array = (tensor.numpy() * 255).astype(np.uint8)
        # 使用 PIL.Image.fromarray 创建 PIL 图像
        return Image.fromarray(array, mode='L')
    else:
        tensor = tensor * 255
        return Image.fromarray(tensor.numpy().astype(np.uint8), mode='RGB')

def pil_to_tensor(image, is_mask=False):
    if image.mode not in ['L', 'RGB']:
        raise ValueError("Image mode must be 'L' or 'RGB'")
    array = np.array(image)
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
    if is_mask:
        # 掩膜添加批次维度，保持形状为 (1, H, W)
        return tensor.unsqueeze(0)
    else:
        # 处理普通图像
        if tensor.dim() == 2:
            # 单通道图像添加通道维度 (H, W) -> (H, W, 1)
            tensor = tensor.unsqueeze(-1)
        # 调整维度顺序为 (批次, 通道, 高, 宽)
        return tensor.unsqueeze(0).permute(0, 3, 1, 2)

def process_and_merge(garment_pil, model_pil, model_garment_pil):
    # 确定目标宽度
    target_width = min(garment_pil.width, model_pil.width)
    
    # 缩放图像
    left_ratio = target_width / garment_pil.width
    left_height = int(garment_pil.height * left_ratio)
    left_resized = garment_pil.resize((target_width, left_height), Image.LANCZOS)
    
    right_ratio = target_width / model_pil.width
    right_height = int(model_pil.height * right_ratio)
    right_resized = model_pil.resize((target_width, right_height), Image.LANCZOS)
    
    # 缩放Mask
    mask_resized = model_garment_pil.resize((target_width, right_height), Image.NEAREST)
    
    # 确定最大高度
    max_height = max(left_height, right_height)
    
    # 创建并填充画布
    canvas = Image.new("RGB", (target_width*2, max_height))
    mask_canvas = Image.new("L", (target_width*2, max_height))
    
    # 合并左图
    canvas.paste(left_resized, (0, (max_height-left_height)//2))
    
    # 合并右图和Mask
    canvas.paste(right_resized, (target_width, (max_height-right_height)//2))
    mask_canvas.paste(mask_resized, (target_width, (max_height-right_height)//2))
    
    return canvas, mask_canvas

class FewBoxOutfit:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "garment": ("IMAGE",),
                "model": ("IMAGE",),
                "model_garment": ("MASK",),
            },
        }
 
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("tryon ", "fit", )
 
    FUNCTION = "convert"
 
    #OUTPUT_NODE = False
 
    CATEGORY = CAPTION.Category
 
    def convert(self, garment, model, model_garment):
        garment_pil = tensor_to_pil(garment)
        #garment_pil.save('D:\\1test.png')
        model_pil = tensor_to_pil(model)
        #model_pil.save('D:\\2test.png')
        model_garment_pil = tensor_to_pil(model_garment, is_mask=True)
        model_garment_pil.save('D:\\3test.png')
        #merged_image, merged_mask = process_and_merge(garment_pil, model_pil, model_garment_pil)
        #merged_image.save('D:\\test1.png')
        #merged_mask.save('D:\\test2.png')
        #tryon = pil_to_tensor(merged_image)
        #fit = pil_to_tensor(merged_mask, is_mask=True)
        #return (tryon, fit)
        return (garment, model_garment)