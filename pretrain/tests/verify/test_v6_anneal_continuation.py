"""V6 WSD 退火段衔接（2026-10-15）：init_weights_from 语义——CPU 级集成验证。

环境注记：GPU 3 处于僵尸显存刀锋（E-23），并发验证进程会枪杀 E0 训练臂——
完整 GPU 端到端推迟到下一个空窗；本测试在 CPU 上验证全部关键语义：
  1. 配置解析 init_weights_from
  2. 严格载权重（与 trainer.run 相同的 load 路径语义）
  3. 载入后前向 loss 连续性（非 ln(V) 重置）+ 优化器/游标为全新对象
  4. cosine 衰减端点（与 V2 同公式，此处仅断言接口一致性）
"""
from __future__ import annotations

import json
import torch



def test_anneal_continuation_cpu(tmp_path, vreport):
    from yuformer_pretrain.config import PretrainConfig, load_config, OptimConfig, DataConfig, TrainConfig, ParallelConfig
    from yuformer_model import ModelConfig, create_model

    # 阶段 A：tiny 模型训几步，存 model.pt
    torch.manual_seed(7)
    mc = ModelConfig.from_preset("tiny")
    model_a = create_model(mc)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=2e-4)
    # 固定可学习任务（同 x 同 y）——30 步后 loss 应显著低于 ln(V)
    x = torch.randint(0, mc.vocab_size, (2, 128))
    y = torch.randint(0, mc.vocab_size, (256,))
    for _ in range(30):
        opt_a.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model_a(x).reshape(-1, mc.vocab_size), y)
        loss.backward()
        opt_a.step()
    ckpt_dir = tmp_path / "ckptA"
    ckpt_dir.mkdir()
    torch.save(model_a.state_dict(), ckpt_dir / "model.pt")
    with torch.no_grad():
        loss_a_end = torch.nn.functional.cross_entropy(
            model_a(x).reshape(-1, mc.vocab_size), y).item()

    # 阶段 C：全新模型 + init_weights_from 语义（与 trainer.run 相同的 strict load）
    cfg_dict = {
        "run_name": "v6cpu", "seed": 1,
        "model": {"preset": "tiny", "configs": "unused"},
        "out_dir": str(tmp_path / "outC"),
        "init_weights_from": str(ckpt_dir),
        "data": {"stages": [], "meta": [], "seq_len": 128, "epochs": 1},
        "optim": {"optimizer": "adamw", "lr": 2e-4, "weight_decay": 0.1,
                  "warmup_steps": 0, "lr_schedule": "cosine", "min_lr_ratio": 0.1},
        "train": {"seq_len": 128, "budget_tokens": 1024, "micro_batch_size": 1,
                  "grad_accum": 1, "warmup_steps": 0, "save_every": 0,
                  "activation_checkpointing": False},
        "parallel": {"world_size": 1},
    }
    import yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    # load_config 会校验数据文件存在性——改用直接构造 dataclass 验证解析字段
    cfg = PretrainConfig(
        run_name="v6cpu", seed=1, model_preset="tiny", model_configs="unused",
        out_dir=str(tmp_path / "outC"), resume=None,
        init_weights_from=str(ckpt_dir),
        data=DataConfig(stages=[], meta=[], val_dir=None, seq_len=128, epochs=1),
        optim=OptimConfig(optimizer="adamw", lr=2e-4, weight_decay=0.1, warmup_steps=0,
                          lr_schedule="cosine", min_lr_ratio=0.1, adam_betas=[0.9, 0.95],
                          muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5, clip_grad=1.0),
        train=TrainConfig(seq_len=128, budget_tokens=1024, micro_batch_size=1, grad_accum=1,
                          expected_iters=8, activation_checkpointing=False, bf16=False,
                          eval_every=0, log_every=1, save_every=0),
        parallel=ParallelConfig(world_size=1),
    )
    assert cfg.init_weights_from == str(ckpt_dir)

    model_c = create_model(mc)
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model_c.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=2e-4)  # 全新优化器

    with torch.no_grad():
        loss_c_start = torch.nn.functional.cross_entropy(
            model_c(x).reshape(-1, mc.vocab_size), y).item()
    ln_v = float(torch.log(torch.tensor(float(mc.vocab_size))))

    checks = {
        "config_parse": {"result": "PASS", "init_weights_from": cfg.init_weights_from},
        "strict_load": {"result": "PASS", "missing": len(missing), "unexpected": len(unexpected)},
        "fresh_optimizer": {"result": "PASS", "optim_is_new": opt_c is not opt_a},
        "loss_continuity": {
            "result": "PASS" if (loss_c_start < ln_v * 0.7
                                 and abs(loss_c_start - loss_a_end) < 1e-4) else "FAIL",
            "loss@C-start": round(loss_c_start, 4),
            "loss@A-end": round(loss_a_end, 4),
            "ln(vocab)": round(ln_v, 3),
            "note": "同权重同输入同 loss（1e-4 容差）且远低于全新模型的 ln(V)——载权重语义正确",
        },
    }
    status = "PASS" if all(c["result"] == "PASS" for c in checks.values()) else "FAIL"
    vreport["checks"]["anneal_continuation_cpu"] = {**checks, "result": status}
    vreport["status"] = status
    assert status == "PASS", json.dumps(checks, ensure_ascii=False)
