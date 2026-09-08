"""Run the official VERL V1 synchronous trainer with GRPO, on two A800 GPUs."""

import argparse
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, type=Path)
p.add_argument("--model", required=True, type=Path)
p.add_argument("--data", required=True, type=Path)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--dry-run", action="store_true")
p.add_argument("--name", required=True)
p.add_argument("--lazy", action="store_true")
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=5282)
p.add_argument("--resume")
p.add_argument("--save-freq", type=int, default=50)
p.add_argument("--audit-checkpoints", action="store_true")
p.add_argument("--timeout-seconds", type=int, default=10800)
p.add_argument("--test-freq", type=int, default=50)
p.add_argument(
    "--val-before-train", action=argparse.BooleanOptionalAction, default=True
)
a = p.parse_args()
scripts = Path(__file__).resolve().parent
out = a.out.resolve()
source = a.source.resolve()
model = a.model.resolve()
data = a.data.resolve()
python = sys.executable
from common import child_environment, preflight, run_logged

if a.steps < 1 or a.timeout_seconds < 1:
    p.error("steps and timeout must be positive")
overrides = [
    "trainer.use_v1=True",
    "trainer.v1.trainer_mode=sync",
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=True",
    f"data.train_files={data}/train.parquet",
    f"data.val_files={data}/test.parquet",
    "data.train_batch_size=8",
    "data.val_batch_size=16",
    "data.max_prompt_length=512",
    "data.max_response_length=512",
    "data.filter_overlong_prompts=True",
    "data.filter_overlong_prompts_workers=1",
    "data.truncation=error",
    "data.shuffle=False",
    f"data.seed={a.seed}",
    "data.dataloader_num_workers=0",
    f"actor_rollout_ref.model.path={model}",
    "actor_rollout_ref.model.external_lib=grpo_probe",
    "actor_rollout_ref.model.use_remove_padding=False",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
    "actor_rollout_ref.actor.strategy=fsdp2",
    "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2",
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
    "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    "actor_rollout_ref.actor.fsdp_config.offload_policy=False",
    f"actor_rollout_ref.actor.fsdp_config.optimizer_offload_step={a.lazy}",
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
    f"actor_rollout_ref.actor.fsdp_config.seed={a.seed}",
    "actor_rollout_ref.actor.fsdp_config.full_determinism=True",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
    "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false,fused:false}",
    "actor_rollout_ref.actor.ppo_mini_batch_size=4",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.actor.ppo_epochs=2",
    "actor_rollout_ref.actor.use_dynamic_bsz=False",
    "actor_rollout_ref.actor.use_torch_compile=False",
    "actor_rollout_ref.actor.shuffle=False",
    f"actor_rollout_ref.actor.data_loader_seed={a.seed}",
    "actor_rollout_ref.actor.use_kl_loss=True",
    "actor_rollout_ref.actor.kl_loss_coef=0.001",
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.ref.strategy=fsdp2",
    "actor_rollout_ref.ref.fsdp_config.strategy=fsdp2",
    "actor_rollout_ref.ref.fsdp_config.fsdp_size=2",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.mode=async",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.n=4",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.25",
    "actor_rollout_ref.rollout.enforce_eager=True",
    "actor_rollout_ref.rollout.free_cache_engine=True",
    "actor_rollout_ref.rollout.max_model_len=1024",
    "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
    "actor_rollout_ref.rollout.max_num_seqs=32",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.full_determinism=True",
    f"actor_rollout_ref.rollout.seed={a.seed}",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
    "actor_rollout_ref.rollout.val_kwargs.n=1",
    "actor_rollout_ref.rollout.agent.num_workers=1",
    "reward.num_workers=1",
    "actor_rollout_ref.rollout.agent.default_agent_loop=grpo_seeded_single_turn",
    f"actor_rollout_ref.rollout.agent.agent_loop_config_path={scripts}/agent_loops.yaml",
    f"reward.custom_reward_function.path={scripts}/grpo_reward.py",
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
    "transfer_queue.backend.SimpleStorage.total_storage_size=2048",
    "trainer.nnodes=1",
    "trainer.n_gpus_per_node=2",
    "trainer.logger=[console,file]",
    "trainer.project_name=verl_grpo_lazy_offload",
    f"trainer.experiment_name={a.name}",
    f"trainer.total_training_steps={a.steps}",
    "trainer.total_epochs=2",
    f"trainer.default_local_dir={out}/checkpoints",
    f"trainer.save_freq={a.save_freq}",
    "trainer.max_actor_ckpt_to_keep=1",
    f"trainer.test_freq={a.test_freq}",
    f"trainer.val_before_train={a.val_before_train}",
    f"trainer.validation_data_dir={out}/validation",
    f"trainer.rollout_data_dir={out}/rollouts",
    "trainer.resume_mode=disable",
    "ray_kwargs.ray_init.num_cpus=16",
    "+ray_kwargs.ray_init.address=local",
    "+ray_kwargs.ray_init.include_dashboard=False",
    "+ray_kwargs.ray_init.object_store_memory=2147483648",
]
if a.resume:
    overrides = [x for x in overrides if not x.startswith("trainer.resume_mode=")]
    overrides += [
        "trainer.resume_mode=resume_path",
        f"trainer.resume_from_path={a.resume}",
    ]
env = dict(
    child_environment(source),
    PYTHONPATH=os.pathsep.join([str(scripts), str(source)]),
    GRPO_AUDIT_DIR=str(out / "audit"),
    GRPO_CHECKPOINT_AUDIT="1" if a.audit_checkpoints else "0",
    VERL_FILE_LOGGER_PATH=str(out / "metrics.jsonl"),
    VERL_USE_EXTERNAL_PLUGINS="none",
    VERL_DISABLE_FLASH_ATTN_CE="1",
    TOKENIZERS_PARALLELISM="false",
    WANDB_MODE="disabled",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PYTHONHASHSEED=str(a.seed),
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    OMP_NUM_THREADS="4",
    RAY_DEDUP_LOGS="0",
    HYDRA_FULL_ERROR="1",
)
# Quote Hydra path values separately from shell/process argument handling.
path_keys = {
    "data.train_files",
    "data.val_files",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "reward.custom_reward_function.path",
    "trainer.default_local_dir",
    "trainer.validation_data_dir",
    "trainer.rollout_data_dir",
    "trainer.resume_from_path",
}
overrides = [
    key + "=" + json.dumps(value) if key in path_keys else item
    for item in overrides
    for key, value in [item.split("=", 1)]
]
command = [python, str(scripts / "grpo_entry.py"), *overrides]
if a.dry_run:
    print(
        json.dumps(
            {
                "command": command,
                "env": {
                    k: env[k]
                    for k in env
                    if k not in os.environ or env[k] != os.environ[k]
                },
            },
            indent=2,
        )
    )
    sys.exit(0)
if (
    not (model / "config.json").is_file()
    or not (data / "train.parquet").is_file()
    or not (data / "test.parquet").is_file()
):
    p.error("model/config.json and data/{train,test}.parquet must exist")
info = preflight(source)
out.mkdir(parents=True, exist_ok=False)
(out / "preflight.json").write_text(json.dumps(info, indent=2))
# The child uses a private Ray session and shuts it down in finally.
from tempfile import TemporaryDirectory

from gpu_monitor import GPUMonitor

with TemporaryDirectory(prefix="verl-review-ray-") as ray_tmp:
    env["VERL_REVIEW_RAY_TMP"] = ray_tmp
    with GPUMonitor(out / "gpu_samples.csv", [g["uuid"] for g in info["gpus"]]):
        code = run_logged(command, source, env, out, a.timeout_seconds)
print(a.name, "EXIT", code, flush=True)
sys.exit(code)
