import torch
import numpy as np
from PIL import Image
from .captions import CAPTION
from .paths import PATH
import os
from webdav3.client import Client
from webdav3.exceptions import WebDavException

def tensor_to_pil(tensor, is_mask=False):
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
    # 检查图像模式
    if image.mode not in ['L', 'RGB']:
        raise ValueError("Image mode must be 'L' (grayscale) or 'RGB'")

    # 将 PIL 图像转换为 NumPy 数组
    array = np.array(image)

    # 转换为 PyTorch 张量，并将值范围从 [0, 255] 转换为 [0, 1]
    tensor = torch.from_numpy(array).float() / 255.0

    if is_mask:
        # 掩码图像是单通道的，形状为 [H, W]
        if image.mode != 'L':
            raise ValueError("Mask image must be in 'L' mode")
        tensor = tensor.unsqueeze(0)  # 添加通道维度，形状变为 [1, H, W]
    else:
        # 普通图像是多通道的，形状为 [H, W, C]
        if image.mode != 'RGB':
            raise ValueError("Image must be in 'RGB' mode")
        #tensor = tensor.permute(2, 0, 1)  # 调整维度顺序为 [C, H, W]
        tensor = tensor.unsqueeze(0)

    return tensor

def compress(pil, scale):
    if scale == 1:
        return pil
    else:
        scale_width = round(pil.width * scale)
        scale_height = round(pil.height * scale)
        return pil.resize((scale_width, scale_height), resample=Image.Resampling.LANCZOS)

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

class FewBoxInContextLora:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "garment": ("IMAGE",),
                "model": ("IMAGE",),
                "model_garment": ("MASK",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
        }
 
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("tryon ", "fit", )
 
    FUNCTION = "convert"
 
    #OUTPUT_NODE = False
 
    CATEGORY = CAPTION.Category
 
    def convert(self, garment, model, model_garment, scale):
        garment_pil = tensor_to_pil(garment)
        #garment_pil.save('D:\\1test.png')
        model_pil = tensor_to_pil(model)
        #model_pil.save('D:\\2test.png')
        model_garment_pil = tensor_to_pil(model_garment, is_mask=True)
        #model_garment_pil.save('D:\\3test.png')
        merged_image, merged_mask = process_and_merge(compress(garment_pil, scale), compress(model_pil, scale), compress(model_garment_pil, scale))
        #merged_image.save('D:\\test1.png')
        #merged_mask.save('D:\\test2.png')
        tryon = pil_to_tensor(merged_image)
        fit = pil_to_tensor(merged_mask, is_mask=True)
        #return (tryon, fit)
        return (tryon, fit)
    
class FewBoxWebDAV:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fitting": ("IMAGE",),
                "host": ("STRING",),
                "username": ("STRING",),
                "password": ("STRING",),
                "protocol": (["http", "https"], {"default": "http"}),
                "port": ("INT", {"default": 80}),
                "path": ("STRING",),
            },
        }
 
    RETURN_TYPES = ()
    RETURN_NAMES = ()
 
    FUNCTION = "upload_webdav"
 
    OUTPUT_NODE = True
 
    CATEGORY = CAPTION.Category
 
    def upload_webdav(self, fitting, host, username, password, protocol, port, path):
        os.makedirs(PATH.Outfit, exist_ok=True)
        file_path = os.path.join(PATH.Outfit, "fitting.png")
        fitting_pil = tensor_to_pil(fitting)
        fitting_pil.save(file_path)
        hostname = f"{protocol}://{host}:{port}"
        options = {
            'webdav_hostname': hostname,
            'webdav_login': username,
            'webdav_password': password
        }
        client = Client(options)
        try:
            client.upload_sync(remote_path=path, local_path=file_path)
            print("File uploaded successfully.")
        except WebDavException as e:
            print(f"Error uploading file: {e}")    
        return ()