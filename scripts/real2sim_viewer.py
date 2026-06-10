import argparse
import json
import socket
import time
from pathlib import Path

import mujoco
import mujoco.viewer


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
    args = parser.parse_args()

    xml_path = Path(args.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    set_home_keyframe(model, data, args.keyframe)

    joint_qpos_addrs = make_joint_qpos_addrs(model)

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
