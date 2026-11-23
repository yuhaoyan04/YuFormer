import copy
import json
from pathlib import Path

import pytest
import yaml

import run_production_canary as canary


ROOT = Path(__file__).resolve().parents[2]


def test_eight_gpu_plan_preserves_global_batch_and_wsd_boundaries():
    cfg = yaml.safe_load((ROOT / "pretrain/configs/pretrain_base.yaml").read_text())
    assert cfg["parallel"]["world_size"] == 8
    assert cfg["train"]["grad_accum"] == 16
    samples = cfg["parallel"]["world_size"] * cfg["train"]["micro_batch_size"] * cfg["train"]["grad_accum"]
    assert samples * cfg["data"]["seq_len"] == 4_194_304
    budget = json.loads((ROOT/'experiments/e4_curriculum/results/effective_total_training_budget.json').read_bytes())
    assert cfg['train']['expected_iters'] == budget['stages'][0]['updates']
    assert cfg['train']['budget_tokens'] == budget['stages'][0]['tokens']
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in (
        ROOT / "pretrain/configs/pretrain_base.env").read_text()
    assert cfg["data"]["stage_sample_limits"][0] == samples * cfg["optim"]["warmup_steps"]
    assert cfg["data"]["stage_sample_limits"][-1] == samples * cfg["optim"]["decay_steps"]
    for path in (ROOT / "pretrain/configs/eight_gpu_probe").glob("*.yaml"):
        candidate = yaml.safe_load(path.read_text())
        assert candidate["parallel"]["replicate_degree"] * candidate["parallel"]["shard_degree"] == 8
        assert candidate["train"]["grad_accum"] == 16


def test_eight_gpu_deadline_includes_reserve_and_rejects_four_gpu_evidence():
    contract = {"tokens_per_step": 4194304, "steps": 200, "world_size": 8,
                "maximum_projected_days": 33, "planned_training_duty_cycle": 0.9}
    result = {"status": "PASS", "hyperparameters": {
        "world_size": 8, "tokens_per_step": 4194304, "trained_steps": 200,
        "trained_tokens": 200 * 4194304},
        "series": {"throughput(tps)": [{"step": i, "value": 47967.8} for i in range(40)]}}
    slow = canary.evaluate_result(result, contract)
    assert slow["projected_training_days_from_p10"] == pytest.approx(30.16105, rel=1e-5)
    assert slow["projected_1t_days_from_p10"] > 33
    assert slow["status"] == "FAIL"
    faster = copy.deepcopy(result)
    for point in faster["series"]["throughput(tps)"]:
        point["value"] = 54000
    assert canary.evaluate_result(faster, contract)["projected_1t_days_from_p10"] < 30
    assert canary.evaluate_result(faster, contract)["status"] == "PASS"
    faster["hyperparameters"]["world_size"] = 4
    assert "world_size_mismatch" in canary.evaluate_result(faster, contract)["issues"]


def test_canary_gates_fixed_full_chain_and_keeps_the_legacy_1t_proxy_visible():
    allocation = json.loads((ROOT/'experiments/e4_curriculum/results/final-measured-total-runtime-execution-plan.json').read_bytes())['proposal']
    contract = {'tokens_per_step':4194304,'steps':200,'world_size':8,
        'maximum_projected_days':33,'planned_training_duty_cycle':.9,
        'full_chain_runtime':{'budget':allocation['proposed_budget'], 'measurements':allocation['measurements']}}
    result = {'status':'PASS','hyperparameters':{'world_size':8,'tokens_per_step':4194304,
        'trained_steps':200,'trained_tokens':200*4194304},
        'series':{'throughput(tps)':[{'step':i,'value':48560.01953125} for i in range(200)]}}
    review = canary.evaluate_result(result,contract)
    assert review['status']=='PASS'
    assert review['runtime_scope']=='TOTAL_4K_16K_64K'
    assert review['projected_1t_days_from_p10']>33
    assert review['projected_total_training_days']==pytest.approx(allocation['total_days'])
    assert len(review['stage_forecasts'])==3
    slow = copy.deepcopy(contract)
    slow['full_chain_runtime']['measurements']['64k-midtrain']['p10_tps_per_device']=1000
    failed=canary.evaluate_result(result,slow)
    assert failed['status']=='FAIL'
    assert 'projected_total_runtime_exceeds_authorized_ceiling' in failed['issues']
    malformed=copy.deepcopy(contract)
    malformed['full_chain_runtime']['measurements'].pop('64k-midtrain')
    with pytest.raises(ValueError,match='does not close'):
        canary.evaluate_result(result,malformed)
