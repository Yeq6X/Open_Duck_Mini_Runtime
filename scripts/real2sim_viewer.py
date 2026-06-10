import argparse
import json
import math
import socket
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_XML_PATH = (
    Path(__file__).resolve().parents[2]
    / "Open_Duck_Playground"
    / "playground"
    / "open_duck_mini_v2"
    / "xmls"
    / "open_duck_mini_v2.xml"
)


def make_joint_qpos_addrs(model):
    addrs = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            addrs[name] = int(model.jnt_qposadr[joint_id])
    return addrs


def find_free_joint_qpos_addr(model):
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id])
    return None


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def quat_from_accel(accelero, accel_sign=-1.0, roll_sign=1.0, pitch_sign=1.0):
    ax, ay, az = [value * accel_sign for value in accelero]
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-6:
        return None

    ax /= norm
    ay /= norm
    az /= norm

    roll = math.atan2(ay, az) * roll_sign
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * pitch_sign
    return quat_from_euler(roll, pitch, 0.0)


def set_home_keyframe(model, data, keyframe_name):
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    except Exception:
        key_id = -1

    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
        if model.nv:
            data.qvel[:] = 0
        mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default=str(DEFAULT_XML_PATH))
    parser.add_argument("--listen_ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--keyframe", default="home")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Seconds before reporting that no packets are arriving",
    )
    parser.add_argument(
        "--use_imu",
        action="store_true",
        help="Apply IMU accelerometer roll/pitch to the MuJoCo floating base",
    )
    parser.add_argument("--roll_sign", type=float, default=1.0)
    parser.add_argument("--pitch_sign", type=float, default=1.0)
    parser.add_argument(
        "--accel_sign",
        type=float,
        default=-1.0,
        help="Set to 1 if the base appears upside down with the default value",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    set_home_keyframe(model, data, args.keyframe)

    joint_qpos_addrs = make_joint_qpos_addrs(model)
    free_joint_qpos_addr = find_free_joint_qpos_addr(model)
    home_base_quat = None
    if free_joint_qpos_addr is not None:
        home_base_quat = np.array(
            data.qpos[free_joint_qpos_addr + 3 : free_joint_qpos_addr + 7],
            dtype=float,
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_ip, args.port))
    sock.setblocking(False)

    print(f"Loaded MuJoCo model: {xml_path}")
    print(f"Listening on UDP {args.listen_ip}:{args.port}")
    print("Waiting for joint packets...")

    last_packet_t = 0.0
    last_warn_t = 0.0
    received_once = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            try:
                while True:
                    packet, _addr = sock.recvfrom(65535)
                    payload = json.loads(packet.decode("utf-8"))
                    joints = payload.get("joints", {})

                    missing = []
                    for name, angle in joints.items():
                        qpos_addr = joint_qpos_addrs.get(name)
                        if qpos_addr is None:
                            missing.append(name)
                            continue
                        data.qpos[qpos_addr] = float(angle)

                    if args.use_imu and free_joint_qpos_addr is not None:
                        imu = payload.get("imu")
                        if imu is not None:
                            accelero = imu.get("accelero")
                            if accelero is not None:
                                tilt_quat = quat_from_accel(
                                    accelero,
                                    accel_sign=args.accel_sign,
                                    roll_sign=args.roll_sign,
                                    pitch_sign=args.pitch_sign,
                                )
                                if tilt_quat is not None:
                                    base_quat = quat_multiply(
                                        home_base_quat,
                                        tilt_quat,
                                    )
                                    base_quat /= np.linalg.norm(base_quat)
                                    data.qpos[
                                        free_joint_qpos_addr
                                        + 3 : free_joint_qpos_addr
                                        + 7
                                    ] = base_quat

                    mujoco.mj_forward(model, data)
                    last_packet_t = time.time()

                    if not received_once:
                        received_once = True
                        print("Receiving joint packets.")
                        if missing:
                            print("Ignored joints not found in model:", ", ".join(missing))

            except BlockingIOError:
                pass

            now = time.time()
            if received_once and now - last_packet_t > args.timeout and now - last_warn_t > args.timeout:
                print(f"No packets for {args.timeout:g}s")
                last_warn_t = now

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
