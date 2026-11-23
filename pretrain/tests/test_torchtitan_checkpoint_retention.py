import ast
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "backends/torchtitan/yuformer_ext/checkpoint.py"


def purge(tmp_path, complete, incomplete=(), keep=2, milestone=0):
    for step in (*complete, *incomplete):
        path = tmp_path / f"step-{step}"
        path.mkdir()
        if step in complete:
            (path / ".metadata").write_bytes(b"complete")
    tree = ast.parse(SOURCE.read_text())
    manager_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "YuFormerCheckpointManager")
    method = next(node for node in manager_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_purge_stale_checkpoints")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    queue = SimpleQueue()
    manager = SimpleNamespace(folder=str(tmp_path), keep_latest_k=keep, keep_every=milestone,
                              purge_queue=queue, _should_purge=lambda: True)
    namespace[method.name](manager)
    result = []
    while not queue.empty():
        result.append(Path(queue.get()).name)
    return result


def test_pending_save_does_not_evict_a_complete_recovery_point(tmp_path):
    assert purge(tmp_path, complete=(1000, 2000), incomplete=(3000,), keep=2) == []


def test_keep_one_preserves_last_complete_checkpoint_during_async_write(tmp_path):
    assert purge(tmp_path, complete=(1000,), incomplete=(2000,), keep=1) == []


def test_partial_artifacts_do_not_consume_retention_or_get_purged(tmp_path):
    assert purge(tmp_path, complete=(1000, 2000), incomplete=(1500,), keep=1) == ["step-1000"]


def test_milestones_survive_alongside_latest_complete_recovery_point(tmp_path):
    assert purge(tmp_path, complete=(1000, 1100, 1200), incomplete=(1300,),
                 keep=1, milestone=1000) == ["step-1100"]
