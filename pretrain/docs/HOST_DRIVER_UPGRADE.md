# TorchTitan host driver upgrade handoff

YuFormer 的当前固定训练栈使用 NVIDIA Linux driver `>=525.60.13`、PyTorch 2.14.0 `cu126`、
CUDA 12.6 和 H800（SM90）。当前节点的 R535 已满足基础门禁，userspace 环境由
`pretrain/backends/torchtitan/prepare_env.sh` 独立准备；驱动升级必须在宿主机维护窗口执行。

## 当前节点快照（2026-10-22 UTC）

- Ubuntu 22.04 / kernel `5.15.0-92-generic`
- NVIDIA kernel module `535.161.08`
- 8×H800
- cu126 环境 preflight、BF16、Triton 编译和 2 卡 NCCL smoke 已通过
- GPU 占用由外部作业动态变化；watcher 在至少 4 卡空闲时启动前置资格测试，并在最终 4→8 reshard 前等待 8 卡同时空闲

容器内安装驱动不会替换宿主机 kernel module。当前基线不要求升级。仅当资格测试证明所需
CUDA 功能无法通过 minor-version compatibility 运行时，宿主机管理员才需要先 drain GPU 作业，
再升级到集群验证的 R580 或更高 datacenter driver 并重启节点。不要在大型数据物化任务的 merge
期间重启；优先等当前臂原子发布，最稳妥的窗口是八臂 tokenization 全部完成以后。

## 维护窗口前

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
uname -a
cat /proc/driver/nvidia/version
dpkg-query -W 'nvidia-*' > /var/tmp/yuformer-nvidia-packages.before.txt
```

只有 compute process 清空且数据流水线处于可恢复边界时才进入升级。驱动包名称、open/proprietary
kernel module 和 Secure Boot 签名策略由宿主机镜像规范决定；最低版本来自
`pretrain/backends/torchtitan/backend.lock.json`，不能仅以“CUDA toolkit 已安装”代替 driver
检查。

## 重启后的验收顺序

```bash
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
bash pretrain/backends/torchtitan/bootstrap_env.sh
```

`bootstrap_env.sh` 会复用已经发布的 userspace 环境，并运行机器可读 preflight。只有
`data/health/torchtitan-preflight.json`（由 readiness watcher 写入）达到 `status=ready`，才允许
启动 qualification。随后按固定 DAG 执行：数值/恢复/reshard qualification → LR bracket →
batch/LR pilot → E1 scaling → E2 architecture → E3 data mix。

升级验收需保留以下证据：宿主机包快照、重启后 `nvidia-smi`、preflight JSON、qualification
identity 与资格测试报告。若 `nvidia-smi`、NVLink/NCCL 或 CUDA 初始化失败，应停止实验并由宿主机
管理员按镜像规范回滚；不得通过放宽 preflight 绕过。

