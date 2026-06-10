import argparse
import os
import time

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    parser.add_argument("--kp", type=int, default=5)
    parser.add_argument("--hold", type=float, default=5.0, help="Hold time in seconds")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(config, usb_port=args.port)
    ids = list(hwi.joints.values())

    try:
        print("Setting low Kp...")
        hwi.io.set_kps(ids, [1] * len(ids))
        hwi.io.set_kds(ids, [0] * len(ids))
        time.sleep(0.5)

        print("Moving to init_pos...")
        hwi.set_position_all(hwi.init_pos)
        time.sleep(3)

        print(f"Setting Kp to {args.kp}...")
        hwi.io.set_kps(ids, [args.kp] * len(ids))

        print(f"Holding {args.hold} seconds...")
        time.sleep(args.hold)

    finally:
        print("Torque off")
        hwi.io.disable_torque(ids)


if __name__ == "__main__":
    main()
