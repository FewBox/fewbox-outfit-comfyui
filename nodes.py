from .captions import CAPTION

class FewBoxOutfit:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            
            
            },
        }
 
    RETURN_TYPES = ()
    RETURN_NAMES = ()
 
    FUNCTION = "test"
 
    #OUTPUT_NODE = False
 
    CATEGORY = CAPTION.Category
 
    def test(self):
        return ()
    
class FewBoxLogo:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            
            
            },
        }
 
    RETURN_TYPES = ()
    RETURN_NAMES = ()
 
    FUNCTION = "generate_logo"
 
    #OUTPUT_NODE = False
 
    CATEGORY = CAPTION.Category
 
    def generate_logo(self):
        return ()