"""Read-only native-DCP generation and NLL using the qualified model kernels."""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def learned_architecture(model):
    return {k:v for k,v in model.items() if k not in {'max_seq_len','selective_recompute'}}


def causal_greedy(model,head,prompt,*,seq_len,max_new_tokens,eos_token_id,device):
    if not prompt or len(prompt)+max_new_tokens>seq_len or max_new_tokens<=0:
        raise ValueError('generation prompt/reserved output exceed the frozen context budget')
    # Padding the FUTURE keeps the compiled decoder shape fixed. Causal attention
    # prevents those positions from affecting any queried prefix position.
    buffer=torch.zeros((1,seq_len),dtype=torch.long,device=device)
    buffer[0,:len(prompt)]=torch.tensor(prompt,dtype=torch.long,device=device)
    generated=[]
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            position=len(prompt)+len(generated)-1
            hidden=model(buffer)
            logits=head(hidden[:,position:position+1])
            if not torch.isfinite(logits).all():raise ValueError('nonfinite generation logits')
            next_token=int(logits[0,0].argmax())
            generated.append(next_token)
            if next_token==eos_token_id:break
            buffer[0,position+1]=next_token
    return generated


class NativeCheckpointEvaluator:
    def __init__(self,*,handoff_file,model_file,preset,seq_len,device,backend='flash_flex_swa',compile_model=True):
        from yuformer_midtrain.checkpoint_contract import verify_parent_handoff
        handoff_file=Path(handoff_file).resolve();handoff=json.loads(handoff_file.read_bytes())
        checkpoint=Path(handoff['checkpoint']);identity_path=checkpoint.parent.parent/'run_identity.json'
        identity=json.loads(identity_path.read_bytes())
        if digest(identity_path)!=handoff['run_identity_sha256']:
            raise ValueError('evaluation checkpoint run identity changed')
        unsigned={k:v for k,v in identity.items() if k!='identity_sha256'}
        if hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=identity['identity_sha256']:
            raise ValueError('evaluation parent identity seal is invalid')
        parent={'path':str(checkpoint),'metadata_sha256':handoff['metadata_sha256'],
            'run_identity_sha256':handoff['run_identity_sha256'],'handoff_file':str(handoff_file),
            'handoff_sha256':digest(handoff_file)}
        verify_parent_handoff(parent)
        config_path=identity_path.with_name('resolved_config.yaml')
        if digest(config_path)!=identity['resolved_config_sha256']:raise ValueError('parent resolved configuration changed')
        config=yaml.safe_load(config_path.read_bytes())
        previous_file=Path(identity['model_configs'])
        if digest(previous_file)!=identity['model_configs_sha256']:raise ValueError('parent learned architecture changed')
        previous=yaml.safe_load(previous_file.read_bytes())['presets'][config['model']['preset']]
        actual=yaml.safe_load(Path(model_file).read_bytes())['presets'][preset]
        if actual['max_seq_len']!=seq_len or learned_architecture(actual)!=learned_architecture(previous):
            raise ValueError('evaluation may change context capacity/recomputation only, not learned architecture or RoPE')
        from yuformer_ext.model import YuFormerTitanModel
        torch.cuda.set_device(device)
        torch.manual_seed(20260926);torch.cuda.manual_seed(20260926)
        torch.use_deterministic_algorithms(True)
        model_config=YuFormerTitanModel.Config(preset=preset,config_path=str(model_file),sdpa_backend=backend)
        with torch.device('meta'):
            model=YuFormerTitanModel(model_config)
        model.to_empty(device=device);model.init_states(buffer_device=device)
        state=model.state_dict()
        dcp.load(state,checkpoint_id=checkpoint,no_dist=True)
        model.load_state_dict(state,strict=True)
        model.eval();model._skip_lm_head=True
        self.head=model.lm_head
        self.model=torch.compile(model,backend='inductor',dynamic=False) if compile_model else model
        self.device,self.seq_len=device,seq_len
        self.eos_token_id=model.model_config.eos_token_id
        self.identity=identity
        self.inputs={str(p):digest(p) for p in [handoff_file,identity_path,config_path,previous_file,Path(model_file)]}

    def guard(self):
        if any(digest(p)!=sha for p,sha in self.inputs.items()):raise ValueError('evaluation checkpoint/config inputs changed')

    def generate(self,prompt_ids,max_new_tokens):
        self.guard()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=causal_greedy(self.model,self.head,prompt_ids,seq_len=self.seq_len,
                max_new_tokens=max_new_tokens,eos_token_id=self.eos_token_id,device=self.device)
        return result

    def nll(self,token_ids,*,target_mask=None):
        if len(token_ids)<2 or len(token_ids)>self.seq_len+1:
            raise ValueError('NLL input must contain2..S+1actual source tokens')
        self.guard();valid=len(token_ids)-1
        inputs=torch.zeros((1,self.seq_len),dtype=torch.long,device=self.device)
        labels=torch.full((1,self.seq_len),-100,dtype=torch.long,device=self.device)
        inputs[0,:valid]=torch.tensor(token_ids[:-1],device=self.device)
        labels[0,:valid]=torch.tensor(token_ids[1:],device=self.device)
        if target_mask is not None:
            if len(target_mask)!=valid:raise ValueError('NLL target mask has another length')
            mask=torch.tensor(target_mask,dtype=torch.bool,device=self.device)
            labels[0,:valid].masked_fill_(~mask,-100)
        target_count=int((labels!=-100).sum())
        if target_count<=0:raise ValueError('NLL case has no scored target tokens')
        total=torch.zeros((),device=self.device,dtype=torch.float64)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=self.model(inputs)
            for start in range(0,self.seq_len,2048):
                end=min(self.seq_len,start+2048)
                if not bool((labels[:,start:end]!=-100).any()):continue
                logits=self.head(hidden[:,start:end]).float()
                value=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                    labels[:,start:end].reshape(-1),ignore_index=-100,reduction='sum')
                if not torch.isfinite(value):raise ValueError('nonfinite heldout NLL')
                total+=value.double()
        return {'nll_sum':float(total),'target_tokens':target_count,'mean_nll':float(total)/target_count}
