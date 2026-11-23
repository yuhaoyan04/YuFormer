"""Adapt the frozen YuFormer tokenizer to RULER's two-method tokenizer API."""
from pathlib import Path

from tokenizers import Tokenizer


class RulerYuFormerTokenizer:
    def __init__(self,model_path):
        path=Path(model_path)
        if path.is_dir():path=path/'tokenizer.json'
        self.backend=Tokenizer.from_file(str(path))

    def text_to_tokens(self,text):
        return self.backend.encode(text,add_special_tokens=False).tokens

    def tokens_to_text(self,tokens):
        return self.backend.decoder.decode(tokens)
