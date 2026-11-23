import pytest
import torch

from yuformer_eval.native_inference import causal_greedy


def test_fixed_future_padding_matches_unpadded_causal_prefix_generation():
    torch.manual_seed(17)
    embedding=torch.nn.Embedding(19,12)
    projection=torch.nn.Linear(12,19,bias=False)
    class Decoder(torch.nn.Module):
        def forward(self,x):
            # Each position depends on the entire preceding prefix and itself.
            return torch.cumsum(embedding(x),dim=1)
    decoder=Decoder();prompt=[3,5,7];expected=[]
    with torch.inference_mode():
        for _ in range(6):
            prefix=torch.tensor([prompt+expected])
            expected.append(int(projection(decoder(prefix)[:,-1:])[0,0].argmax()))
    actual=causal_greedy(decoder,projection,prompt,seq_len=16,max_new_tokens=6,eos_token_id=-1,device='cpu')
    assert actual==expected


def test_generation_refuses_to_silently_truncate_a_long_prompt():
    with pytest.raises(ValueError,match='context budget'):
        causal_greedy(None,None,[1]*10,seq_len=12,max_new_tokens=3,eos_token_id=2,device='cpu')
