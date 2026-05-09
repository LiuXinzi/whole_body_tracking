import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        try:
            # 尝试多种方式检测是否使用 wandb
            if wandb.run is not None:
                policy_path = path.split("model")[0]
                filename = policy_path.split("/")[-2] + ".onnx"
                export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        except Exception:
            # 如果任何步骤失败，就跳过 onnx 导出部分，不影响训练保存
            pass


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        try:
            # 尝试多种方式检测是否使用 wandb
            if wandb.run is not None:
                policy_path = path.split("model")[0]
                filename = policy_path.split("/")[-2] + ".onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

                # link the artifact registry to this run
                if self.registry_name is not None:
                    wandb.run.use_artifact(self.registry_name)
                    self.registry_name = None
        except Exception:
            # 如果任何步骤失败，就跳过 onnx 导出部分，不影响训练保存
            pass