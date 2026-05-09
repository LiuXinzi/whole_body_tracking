from isaaclab.utils import configclass

from whole_body_tracking.robots.men import MEN_ACTION_SCALE, MEN_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class MenFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = MEN_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_yaw_Link"
        self.commands.motion.body_names = [
            "pelvis_link",
            "left_hip_roll_Link",
            "left_knee_Link",
            "left_ankle_roll_Link",
            "right_hip_roll_Link",
            "right_knee_Link",
            "right_ankle_roll_Link",
            "torso_yaw_Link",
            "left_shoulder_roll_Link",
            "left_elbow_Link",
            "left_wrist_yaw_Link",
            "right_shoulder_roll_Link",
            "right_elbow_Link",
            "right_wrist_yaw_Link",
        ]