import torch
import numpy as np
from PIL import Image
from .captions import CAPTION
from .paths import PATH
import os
from webdav3.client import Client
from webdav3.exceptions import WebDavException

def image_tensor_to_pil(tensor):
    # 去掉Batch维度（[B,H,W,C] -> [H,W,C]）
    # [1, 1785, 1340, 3] -> [1785, 1340, 3]
    tensor = tensor.squeeze(0)
    tensor = tensor * 255
    return Image.fromarray(tensor.numpy().astype(np.uint8), mode='RGB')
        
    
def mask_tensor_to_pil(tensor):
    # 去掉Batch维度（[B,H,W] -> [H,W]）
    # [1, 1785, 1340] -> [1785, 1340]
    tensor = tensor.squeeze(0)
    tensor = tensor * 255
    return Image.fromarray(tensor.numpy().astype(np.uint8), mode='L')

def mask_tensor_to_transparent_pil(tensor):
    # 去掉Batch维度（[B,H,W] -> [H,W]）
    # [1, 1785, 1340] -> [1785, 1340]
    tensor = tensor.squeeze(0)
    # 增加Channel维度（[H,W] -> [H,W,C]）
    # [1785, 1340] -> [1785, 1340, 1]
    tensor = tensor.unsqueeze(2)
    # [1785, 1340, 1] -> [1785, 1340, 2]
    tensor = torch.cat([tensor, 1 - tensor], dim=2)
    tensor = tensor * 255
    return Image.fromarray(tensor.numpy().astype(np.uint8), mode='LA')

def mask_tensor_to_revert_transparent_pil(tensor):
    # 去掉Batch维度（[B,H,W] -> [H,W]）
    # [1, 1785, 1340] -> [1785, 1340]
    tensor = tensor.squeeze(0)
    # 去掉Channel维度（[H,W] -> [H,W,C]）
    # [1785, 1340] -> [1785, 1340, 1]
    tensor = tensor.unsqueeze(2)
    # [1785, 1340, 1] -> [1785, 1340, 2]
    tensor = tensor.repeat(1, 1, 2)
    tensor = tensor * 255
    return Image.fromarray(tensor.numpy().astype(np.uint8), mode='LA')

def image_pil_to_tensor(image):
    # 检查图像模式
    if image.mode not in ['RGB', 'RGBA']:
        raise ValueError(f"{image.mode} Image mode must be 'RGB' or 'RGBA'")
    array = np.array(image.convert("RGB"))
    tensor = torch.from_numpy(array).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor

def mask_pil_to_tensor(image):
    # 检查图像模式
    if image.mode not in ['L', 'LA']:
        raise ValueError("Image mode must be 'L' (grayscale) or 'RGB'")

    # 将 PIL 图像转换为 NumPy 数组
    array = np.array(image)

    # 转换为 PyTorch 张量，并将值范围从 [0, 255] 转换为 [0, 1]
    tensor = torch.from_numpy(array).float() / 255.0

    if image.mode not in ['L', 'LA']:
        raise ValueError("Mask image must be in 'L' 'LA' mode")
    # 掩码图像是单通道的，形状为 [H, W]
    if image.mode == 'L':
        tensor = tensor.unsqueeze(0)  # 添加通道维度，形状变为 [1, H, W]
    # TODO: 掩码图像是包含Alpha通道，形状为 [H, W, C]
    elif image.mode == 'LA':
        tensor = tensor.mean(-1)
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
    garment_spect_ratio = garment_pil.width / garment_pil.height
    model_spect_ratio = model_pil.width / model_pil.height
    
    # 缩放Garment图像宽度匹配Model
    if garment_spect_ratio > model_spect_ratio:
        left_resized = garment_pil.resize((model_pil.width, round(model_pil.width * (garment_pil.height / garment_pil.width))), Image.LANCZOS)
    else:
        left_resized = garment_pil.resize((round(model_pil.height * (garment_pil.width / garment_pil.height)), model_pil.height), Image.LANCZOS)
    # 创建并填充画布
    canvas = Image.new("RGB", (model_pil.width*2, model_pil.height))
    mask_canvas = Image.new("L", (model_pil.width*2, model_pil.height))
    
    # 合并左图
    if garment_spect_ratio > model_spect_ratio:
        canvas.paste(left_resized, (0, (model_pil.height-left_resized.height)//2))
    else:
        canvas.paste(left_resized, ((model_pil.width-left_resized.width)//2, 0))
    
    # 合并右图和Mask
    canvas.paste(model_pil, (model_pil.width, 0))
    mask_canvas.paste(model_garment_pil, (model_pil.width, 0))
    
    return canvas, mask_canvas


def open_image(path):
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        print(f"Unable to load image: {e}")
        return None

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
        garment_pil = image_tensor_to_pil(garment)
        #garment_pil.save('D:\\1test.png')
        model_pil = image_tensor_to_pil(model)
        #model_pil.save('D:\\2test.png')
        model_garment_pil = mask_tensor_to_pil(model_garment)
        #model_garment_pil.save('D:\\3test.png')
        merged_image, merged_mask = process_and_merge(compress(garment_pil, scale), compress(model_pil, scale), compress(model_garment_pil, scale))
        #merged_image.save('D:\\test1.png')
        #merged_mask.save('D:\\test2.png')
        tryon = image_pil_to_tensor(merged_image)
        fit = mask_pil_to_tensor(merged_mask)
        #return (pil_to_tensor(garment_pil), fit)
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
        fitting_pil = image_tensor_to_pil(fitting)
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
    
class FewBoxWatermark:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("IMAGE",),
                "watermark_path": ("STRING",),
                "scale": ([0.1, 0.15, 0.2, 0.25], {"default": 0.1}),
                "margin": ([10, 15, 20], {"default": 10}),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reproduction",)
 
    FUNCTION = "embed"
 
    OUTPUT_NODE = True
 
    CATEGORY = CAPTION.Category
 
    def embed(self, original, watermark_path, scale, margin):
        original_pil = image_tensor_to_pil(original)
        watermark_pil = open_image(watermark_path)
        scale_ratio = min(int(original_pil.width * scale) / watermark_pil.width, int(original_pil.height * scale) / watermark_pil.height)
        watermark_pil = compress(watermark_pil, scale_ratio)
        position = (original_pil.width - watermark_pil.width - margin, original_pil.height - watermark_pil.height - margin)
        combined = Image.new("RGBA", original_pil.size, (0, 0, 0, 0))
        combined.paste(original_pil, (0, 0))
        combined.paste(watermark_pil, position, mask=watermark_pil)
        #file_path = os.path.join(PATH.Lab, "watermark.png")
        #combined.save(file_path)
        return (image_pil_to_tensor(combined),)
    
class FewBoxSaveImage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fitting": ("IMAGE",),
                "name": ("STRING",),
            },
        }
 
    RETURN_TYPES = ()
    RETURN_NAMES = ()
 
    FUNCTION = "save"
 
    OUTPUT_NODE = True
 
    CATEGORY = CAPTION.Category
 
    def save(self, fitting, name):
        os.makedirs(PATH.Output, exist_ok=True)
        file_path = os.path.join(PATH.Output, name)
        fitting_pil = image_tensor_to_pil(fitting)
        fitting_pil.save(file_path)
        return ()

class FewBoxLab:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "original_mask": ("MASK",)
            },
            "optional": {},
            "hidden": {}
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
 
    FUNCTION = "experiment"
 
    #OUTPUT_NODE = True
 
    CATEGORY = CAPTION.Category
 
    def experiment(self, original_image, original_mask):
        print(f"Original Image Shape: {original_image.shape}")
        original_image_pil = image_tensor_to_pil(original_image)
        original_image_path = os.path.join(PATH.Lab, "original_image.png")
        original_image_pil.save(original_image_path)
        print(f"Original Mask Shape: {original_mask.shape}")
        original_mask_pil = mask_tensor_to_pil(original_mask)
        original_mask_path = os.path.join(PATH.Lab, "original_mask.png")
        original_mask_pil.save(original_mask_path)
        original_mask_transparent_pil = mask_tensor_to_transparent_pil(original_mask)
        original_mask_transparent_path = os.path.join(PATH.Lab, "original_mask_transparent.png")
        original_mask_transparent_pil.save(original_mask_transparent_path)
        original_mask_revert_transparent_pil = mask_tensor_to_revert_transparent_pil(original_mask)
        original_mask_revert_transparent_path = os.path.join(PATH.Lab, "original_mask_revert_transparent.png")
        original_mask_revert_transparent_pil.save(original_mask_revert_transparent_path)
        return ('Go FewBox!',)