import argparse
import pickle
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert Men robot pkl motion data to npz")
parser.add_argument("--input_file", type=str, required=True, help="Path to input pkl file")
parser.add_argument("--output_name", type=str, required=True, help="Name for output artifact")
parser.add_argument("--sequence_key", type=str, default=None, help="Sequence key in pkl (default: first key)")
parser.add_argument("--output_fps", type=int, default=50, help="Output fps (default: 50)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_conjugate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source" / "whole_body_tracking"))
from whole_body_tracking.robots.men import MEN_CFG, MEN_ACTUATED_JOINT_ORDER


# PKL 关节顺序定义（来自 men_pkl_format.txt）
MEN_PKL_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = MEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def interpolate_motion(data, input_fps, output_fps):
    """线性插值运动数据到目标帧率"""
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (data.shape[0] - 1) * input_dt
    num_output_frames = int(duration / output_dt) + 1
    
    output_times = np.linspace(0, duration, num_output_frames)
    input_indices = (output_times / input_dt).astype(int)
    input_indices = np.clip(input_indices, 0, data.shape[0] - 2)
    alpha = (output_times - input_indices * input_dt) / input_dt
    
    alpha = alpha[:, np.newaxis]
    output_data = data[input_indices] * (1 - alpha) + data[input_indices + 1] * alpha
    
    return output_data


def main():
    input_path = Path(args_cli.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print(f"Available pkl files in data_pkl directory:")
        data_pkl_dir = Path(__file__).parent.parent / "source" / "whole_body_tracking" / "New_robot" / "data_pkl"
        for f in sorted(data_pkl_dir.glob("*.pkl")):
            print(f"  - {f.name}")
        return
    
    print(f"Loading pkl file: {input_path}")
    with open(input_path, "rb") as f:
        pkl_data = pickle.load(f)
    
    if args_cli.sequence_key is None:
        sequence_key = sorted(pkl_data.keys())[0]
    else:
        sequence_key = args_cli.sequence_key
    
    print(f"Using sequence key: {sequence_key}")
    sequence = pkl_data[sequence_key]
    
    fps = sequence.get("fps", 60)
    print(f"Input fps: {fps}")
    
    dof = sequence["dof"]
    dof_vel = sequence["dof_vel"]
    root_trans = sequence["root_trans_offset"]
    root_rot = sequence["root_rot"]
    
    print(f"Loaded motion: {dof.shape[0]} frames, {dof.shape[1]} dofs")
    
    if args_cli.output_fps != fps:
        print(f"Interpolating to {args_cli.output_fps} fps")
        dof = interpolate_motion(dof, fps, args_cli.output_fps)
        dof_vel = interpolate_motion(dof_vel, fps, args_cli.output_fps)
        root_trans = interpolate_motion(root_trans, fps, args_cli.output_fps)
        root_rot = interpolate_motion(root_rot, fps, args_cli.output_fps)
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    robot = scene["robot"]
    
    joint_indices = robot.find_joints(MEN_ACTUATED_JOINT_ORDER, preserve_order=True)[0]
    print(f"Joint indices: {joint_indices}")
    
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    
    num_frames = dof.shape[0]
    print(f"Replaying {num_frames} frames...")
    
    for i in range(num_frames):
        root_pos = root_trans[i]
        root_quat_xyzw = root_rot[i]
        root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])
        
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = torch.from_numpy(root_pos).float().to(sim.device)
        root_states[:, 3:7] = torch.from_numpy(root_quat_wxyz).float().to(sim.device)
        robot.write_root_state_to_sim(root_states)
        
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[0, joint_indices] = torch.from_numpy(dof[i]).float().to(sim.device)
        joint_vel[0, joint_indices] = torch.from_numpy(dof_vel[i]).float().to(sim.device)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        sim.render()
        scene.update(sim.get_physics_dt())
        
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
    
    for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        log[k] = np.stack(log[k], axis=0)
    
    output_npz = Path("/tmp") / f"{args_cli.output_name}.npz"
    np.savez(output_npz, **log)
    print(f"Saved npz to: {output_npz}")
    
    import wandb
    COLLECTION = args_cli.output_name
    run = wandb.init(project="csv_to_npz", name=COLLECTION)
    print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
    REGISTRY = "motions"
    logged_artifact = run.log_artifact(artifact_or_path=str(output_npz), name=COLLECTION, type=REGISTRY)
    print(f"[INFO]: Motion saved to wandb: {COLLECTION}")


if __name__ == "__main__":
    main()
    simulation_app.close()