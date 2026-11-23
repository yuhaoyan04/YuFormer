# RL source provenance

YuFormer 的 RL 正式主线固定为 **verl + SGLang**。确切版本写在
[`backend.lock.json`](../backend.lock.json)，禁止使用浮动的 `main`、`latest` 或 `dev` 镜像。

- 新工程轨固定 verl v0.9.1；其官方依赖栈使用 PyTorch 2.13，故必须与 TorchTitan 预训练
  环境分开构建和记录；
- `verl-recipe` commit 固定配方本身，并同时记录 `REQUIRED_VERL.txt` 当前 pin 与论文复现
  所用的旧 verl commit，避免把
  “严格复现 DAPO”与“在较新 verl 上实现同类训练”混为一谈；
- rollout 后端只保留 SGLang v0.5.20 一条正式主线，减少 vLLM/SGLang 双栈漂移。

本目录不 vendoring 上游源码。后续 adapter、配置和 launcher 以独立文件接入，上游 checkout
路径经环境变量提供；每次训练 attempt 都必须记录三个 commit、容器 digest、模型与 tokenizer
hash、SFT checkpoint identity、prompt release、verifier release 和 rollout 参数。

参考：

- https://github.com/verl-project/verl-recipe/tree/main/dapo
- https://github.com/verl-project/verl/releases/tag/v0.9.1
- https://github.com/verl-project/verl/blob/main/docs/perf/best_practices.rst
- https://github.com/sgl-project/sglang/releases/tag/v0.5.20
