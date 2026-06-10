import argparse
import json
import os
import sys
import socket
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_bdx_runtime"))

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ip", required=True, help="Receiver PC IP address")
    parser.add_argument("--port", type=int, default=5005, help="Receiver UDP port")
    parser.add_argument("--serial_port", default="/dev/ttyACM0", help="Motor controller port")
    parser.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--freq", type=float, default=30.0, help="Streaming frequency in Hz")
    parser.add_argument(
        "--disable_torque",
        action="store_true",
        help="Disable all motor torque before streaming positions",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Stream raw servo positions instead of offset-corrected model angles",
    )
    args = parser.parse_args()

    duck_config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(duck_config, usb_port=args.serial_port)
    joint_names = list(hwi.joints.keys())
    joint_ids = list(hwi.joints.values())

    if args.disable_torque:
        print("Disabling torque on all motors...")
        hwi.io.disable_torque(joint_ids)
        time.sleep(0.2)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.target_ip, args.port)
    period = 1.0 / args.freq

    mode = "raw servo positions" if args.raw else "offset-corrected model angles"
    print(f"Streaming {mode} to {args.target_ip}:{args.port} at {args.freq:g} Hz")

    try:
        while True:
            start = time.time()

            if args.raw:
                positions = hwi.io.read_present_position(joint_ids)
            else:
                positions = hwi.get_present_positions()
                if positions is None:
                    time.sleep(period)
                    continue

            payload = {
                "timestamp": start,
                "raw": args.raw,
                "joints": {
                    name: float(position)
                    for name, position in zip(joint_names, positions)
                },
            }

            sock.sendto(json.dumps(payload).encode("utf-8"), target)
            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
