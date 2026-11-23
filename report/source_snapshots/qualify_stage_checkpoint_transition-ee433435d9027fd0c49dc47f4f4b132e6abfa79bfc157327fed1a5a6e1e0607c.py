"""Exercise real shared DCP weight-only initialization and full local CPU resume."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from torchtitan.components.optimizer import OptimizersContainer, LRSchedulersContainer, default_adamw
from yuformer_ext.checkpoint import YuFormerCheckpointManager


class Cursor:
    def __init__(self, seq):
        self.seq, self.consumed = seq, 0

    def state_dict(self):
        return {'seq_len': self.seq, 'consumed_samples': self.consumed}

    def load_state_dict(self, state):
        if state['seq_len'] != self.seq:
            raise ValueError('a stage cursor cannot load another context length')
        self.consumed = state['consumed_samples']


class TrainingState:
    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {'step': self.step, 'rng': torch.get_rng_state()}

    def load_state_dict(self, state):
        self.step = state['step']
        torch.set_rng_state(state['rng'])


def components(folder, *, seq, seed, initial=None):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 4)
    optimizer = default_adamw(lr=.001)
    optimizer.implementation = 'for-loop'
    optimizer = OptimizersContainer(optimizer, model_parts=[model])
    scheduler = LRSchedulersContainer.Config(warmup_steps=1, total_steps=16,
        min_lr_factor=.1).build(optimizers=optimizer, training_steps=16)
    cursor, state = Cursor(seq), TrainingState()
    config = YuFormerCheckpointManager.Config(enable=True, interval=2, async_mode='disabled',
        keep_latest_k=0, last_save_model_only=False, initial_load_path=str(initial) if initial else None,
        initial_load_model_only=True)
    checkpoint = YuFormerCheckpointManager(config, dataloader=cursor, model_parts=[model],
        optimizers=optimizer, lr_schedulers=scheduler, states={'train_state': state},
        sd_adapter=None, base_folder=str(folder))
    return model, optimizer, scheduler, cursor, state, checkpoint


def advance(values, steps):
    model, optimizer, scheduler, cursor, state, _ = values
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(2, 4)).square().mean().backward()
        optimizer.step()
        scheduler.step()
        state.step += 1
        cursor.consumed += 8


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def qualify(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve previous checkpoint-transition qualification attempts')
    output.mkdir(parents=True)
    report = {'status': 'RUNNING_CPU_STAGE_TRANSITION', 'production_qualified': False,
              'gpu_stage_transition_still_required': True, 'checks': {}}
    try:
        parent = components(output/'base-4k', seq=4096, seed=11)
        advance(parent, 4)
        parent[5].save(4, last_step=False)
        parent_path = output/'base-4k/checkpoint/step-4'
        parent_weights = copy.deepcopy(parent[0].state_dict())
        stage = components(output/'stage-16k', seq=16384, seed=22, initial=parent_path)
        initial_rng = torch.get_rng_state().clone()
        if not stage[5].load():
            raise ValueError('initial parent checkpoint was not loaded')
        assert same(parent_weights, stage[0].state_dict())
        assert not stage[1].state_dict()['state']
        assert stage[2].state_dict()['last_epoch'] == 0
        assert stage[3].consumed == 0 and stage[3].seq == 16384 and stage[4].step == 0
        assert torch.equal(initial_rng, torch.get_rng_state())
        report['checks']['weight_only_preserves_new_optimizer_scheduler_cursor_rng'] = 'PASS'
        advance(stage, 2)
        stage[5].save(2, last_step=False)
        saved = [copy.deepcopy(value.state_dict()) for value in stage[:5]]
        # Keep the same initial_* settings: the valid LOCAL checkpoint must win.
        resumed = components(output/'stage-16k', seq=16384, seed=99, initial=parent_path)
        assert resumed[5].load()
        assert all(same(expected, actual.state_dict()) for expected, actual in zip(saved, resumed[:5]))
        assert resumed[4].step == 2 and resumed[3].consumed == 16
        report['checks']['local_resume_restores_all_states_while_initial_model_only_remains_true'] = 'PASS'
        # The next actual optimizer update must equal the uninterrupted update.
        advance(resumed, 1)
        after_resume = [copy.deepcopy(value.state_dict()) for value in resumed[:5]]
        stage[4].load_state_dict(saved[4])
        advance(stage, 1)
        assert all(same(expected, actual.state_dict()) for expected, actual in zip(after_resume, stage[:5]))
        report['checks']['next_optimizer_update_and_rng_match_uninterrupted'] = 'PASS'
        assert not torch.cuda.is_initialized()
        report.update(status='PASS_ACTUAL_CPU_DCP_STAGE_TRANSITION_NOT_GPU_QUALIFICATION',
            torch_version=torch.__version__, parent_metadata_sha256=hashlib.sha256((parent_path/'.metadata').read_bytes()).hexdigest(),
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for values in (parent, stage, resumed):
            values[5].close()
    except Exception as exc:
        report.update(status='FAIL_CPU_STAGE_TRANSITION', error=repr(exc))
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('this CPU qualification requires explicit CUDA_VISIBLE_DEVICES empty')
    print(json.dumps(qualify(args.output), sort_keys=True))
