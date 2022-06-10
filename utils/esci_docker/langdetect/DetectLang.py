from typing import List


class DetectLang:
    def __init__(self,model_path=None,**kwargs) -> None:
        self.module_path = model_path
    
    def detect_lang(self,text:List[str],batch_size) -> List[str]:
        raise NotImplementedError

