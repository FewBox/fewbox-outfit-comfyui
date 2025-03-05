from .captions import CAPTION
from .nodes import FewBoxInContextLora, FewBoxLab, FewBoxSaveImage, FewBoxWatermark, FewBoxWebDAV

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique

NODE_CLASS_MAPPINGS = {
    "FewBoxInContextLora": FewBoxInContextLora,
    "FewBoxWebDAV": FewBoxWebDAV,
    "FewBoxWatermark": FewBoxWatermark,
    "FewBoxSaveImage": FewBoxSaveImage,
    "FewBoxLab": FewBoxLab
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FewBoxInContextLora": CAPTION.InContextLora,
    "FewBoxWebDAV": CAPTION.WebDAV,
    "FewBoxWatermark": CAPTION.Watermark,
    "FewBoxSaveImage": CAPTION.SaveImage,
    "FewBoxLab": CAPTION.Lab
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]