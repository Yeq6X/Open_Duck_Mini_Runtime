import argparse
import json
import math
import socket
import threading
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

DEFAULT_OFFSETS_PATH = Path("real2sim_offsets.json")


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


def load_offsets(path, joint_names):
    offsets = {name: 0.0 for name in joint_names}
    if path is None:
        return offsets

    path = Path(path).expanduser()
    if not path.exists():
        return offsets

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    values = data.get("joints_offsets", data)
    for name in joint_names:
        if name in values:
            offsets[name] = float(values[name])
    return offsets


def save_offsets(path, offsets, joint_names):
    path = Path(path).expanduser()
    payload = {
        "joints_offsets": {
            name: float(offsets.get(name, 0.0))
            for name in joint_names
        }
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def start_tuner_gui(offsets, latest_values, joint_names, lock, save_path):
    import tkinter as tk
    from tkinter import ttk

    def run():
        root = tk.Tk()
        root.title("Real2Sim Offset Tuner")
        root.geometry("420x230")

        selected_joint = tk.StringVar(value=joint_names[0])
        step_value = tk.StringVar(value="0.01")
        offset_value = tk.StringVar()
        raw_value = tk.StringVar()
        model_value = tk.StringVar()
        status_value = tk.StringVar(value="Use raw joint streaming on the robot.")

        def get_step():
            try:
                return float(step_value.get())
            except ValueError:
                status_value.set("Invalid step value")
                return 0.0

        def adjust(sign):
            joint = selected_joint.get()
            step = get_step()
            with lock:
                offsets[joint] = float(offsets.get(joint, 0.0)) + sign * step
            refresh()

        def zero_selected():
            joint = selected_joint.get()
            with lock:
                offsets[joint] = 0.0
            refresh()

        def print_offsets():
            with lock:
                print(json.dumps({"joints_offsets": offsets}, indent=4))

        def save():
            with lock:
                save_offsets(save_path, offsets, joint_names)
            status_value.set(f"Saved {save_path}")

        def refresh():
            joint = selected_joint.get()
            with lock:
                offset = offsets.get(joint, 0.0)
                raw = latest_values["raw"].get(joint)
                model = latest_values["model"].get(joint)

            offset_value.set(f"{offset:+.6f} rad")
            raw_value.set("---" if raw is None else f"{raw:+.6f} rad")
            model_value.set("---" if model is None else f"{model:+.6f} rad")
            root.after(100, refresh)

        main_frame = ttk.Frame(root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Joint").grid(row=0, column=0, sticky="w")
        joint_box = ttk.Combobox(
            main_frame,
            textvariable=selected_joint,
            values=joint_names,
            state="readonly",
            width=28,
        )
        joint_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=4)

        ttk.Label(main_frame, text="Offset").grid(row=1, column=0, sticky="w")
        ttk.Label(main_frame, textvariable=offset_value).grid(row=1, column=1, sticky="w")
        ttk.Label(main_frame, text="Raw").grid(row=2, column=0, sticky="w")
        ttk.Label(main_frame, textvariable=raw_value).grid(row=2, column=1, sticky="w")
        ttk.Label(main_frame, text="Model").grid(row=3, column=0, sticky="w")
        ttk.Label(main_frame, textvariable=model_value).grid(row=3, column=1, sticky="w")

        ttk.Label(main_frame, text="Step").grid(row=4, column=0, sticky="w")
        ttk.Entry(main_frame, textvariable=step_value, width=10).grid(
            row=4, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Button(main_frame, text="-", command=lambda: adjust(-1.0)).grid(
            row=5, column=0, sticky="ew", padx=3, pady=6
        )
        ttk.Button(main_frame, text="+", command=lambda: adjust(1.0)).grid(
            row=5, column=1, sticky="ew", padx=3, pady=6
        )
        ttk.Button(main_frame, text="Zero", command=zero_selected).grid(
            row=5, column=2, sticky="ew", padx=3, pady=6
        )
        ttk.Button(main_frame, text="Print", command=print_offsets).grid(
            row=6, column=0, sticky="ew", padx=3
        )
        ttk.Button(main_frame, text="Save", command=save).grid(
            row=6, column=1, sticky="ew", padx=3
        )

        ttk.Label(main_frame, textvariable=status_value).grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(12, 0)
        )

        main_frame.columnconfigure(1, weight=1)
        refresh()
        root.mainloop()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


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
    parser.add_argument(
        "--offsets_path",
        default=None,
        help="JSON file containing joints_offsets, used when raw positions are received",
    )
    parser.add_argument(
        "--duck_config_path",
        default=None,
        help="duck_config.json to use as initial offset values",
    )
    parser.add_argument(
        "--save_offsets_path",
        default=str(DEFAULT_OFFSETS_PATH),
        help="Where the tuner saves adjusted joints_offsets",
    )
    parser.add_argument(
        "--tuner_gui",
        action="store_true",
        help="Open a small Tkinter GUI for live offset tuning",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    set_home_keyframe(model, data, args.keyframe)

    joint_qpos_addrs = make_joint_qpos_addrs(model)
    joint_names = list(joint_qpos_addrs.keys())
    initial_offsets_path = args.offsets_path or args.duck_config_path
    offsets = load_offsets(initial_offsets_path, joint_names)
    latest_values = {
        "raw": {},
        "model": {},
    }
    values_lock = threading.Lock()
    if args.tuner_gui:
        start_tuner_gui(
            offsets,
            latest_values,
            joint_names,
            values_lock,
            Path(args.save_offsets_path),
        )
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
                    raw_packet = bool(payload.get("raw", False))
                    for name, angle in joints.items():
                        qpos_addr = joint_qpos_addrs.get(name)
                        if qpos_addr is None:
                            missing.append(name)
                            continue
                        raw_angle = float(angle)
                        with values_lock:
                            offset = offsets.get(name, 0.0)
                        model_angle = raw_angle - offset if raw_packet else raw_angle
                        data.qpos[qpos_addr] = model_angle
                        with values_lock:
                            if raw_packet:
                                latest_values["raw"][name] = raw_angle
                            latest_values["model"][name] = model_angle

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
