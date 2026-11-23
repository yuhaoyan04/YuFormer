"""Measure final-checkpoint validation through the actual full-state resume path."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'pretrain/backends/torchtitan'),str(ROOT/'model/src'),
              str(ROOT/'experiments/e4_curriculum/scripts')]
from yuformer_ext.config_registry import yuformer_from_yaml
from torchtitan.tools.logging import init_logger
from torchtitan.observability import structured_logger as sl
from frozen_training_evidence import publish_checked


def main():
    init_logger()
    config=yuformer_from_yaml()
    source=ROOT/'data/experiments/rope_source_controls_20261114/larger_lr1e4_b8'
    checkpoint=source/'checkpoint/step-200'
    config.checkpoint.initial_load_path=str(checkpoint)
    config.checkpoint.initial_load_model_only=False
    config.checkpoint.folder='unused-full-state-resume-diagnostic'
    config.checkpoint.async_mode='disabled'
    sl.init_structured_logger(source='full_checkpoint_resume_diagnostic',output_dir=config.dump_folder,enable=False)
    trainer=None
    measured={}
    try:
        trainer=config.build()
        original_load=trainer.checkpointer.load
        original_log=trainer.metrics_processor.log_validation

        def load(*args,**kwargs):
            result=original_load(*args,**kwargs)
            if not result or trainer.step!=200 or trainer.dataloader.state_dict()['consumed_samples']!=204800:
                raise ValueError('actual full step-200 state was not restored')
            return result

        def log(*args,**kwargs):
            measured.update(aggregate=kwargs['loss'],domains=kwargs['extra_metrics'])
            return original_log(*args,**kwargs)

        def forbid_step(*args,**kwargs):
            raise ValueError('final-checkpoint diagnostic must execute zero optimizer updates')

        trainer.checkpointer.load=load
        trainer.metrics_processor.log_validation=log
        trainer.optimizers.step=forbid_step
        trainer.train()
        primary=json.loads((source/'result.json').read_bytes())
        deltas={tag:dict(original=primary['metrics'][tag]['last'],resumed=value,
                        delta=value-primary['metrics'][tag]['last'])
                for tag,value in measured['domains'].items()}
        if dist.get_rank()==0:
            result=dict(status='MEASURED_FULL_STATE_FINAL_RESUME_DIAGNOSTIC_NOT_QUALITY_ADMISSION',
                        checkpoint_metadata_sha256=hashlib.sha256((checkpoint/'.metadata').read_bytes()).hexdigest(),
                        training_steps_executed=0,restored_step=trainer.step,measured=measured,comparisons=deltas,
                        maximum_primary_absolute_delta=max(abs(row['delta']) for row in deltas.values()),
                        production_qualified=False)
            publish_checked(Path(config.dump_folder)/'diagnostic.json',
                            (json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
            print(json.dumps(result,sort_keys=True),flush=True)
    finally:
        if trainer is not None:trainer.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
