import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_bdx_runtime"))

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=None, ignore_default=True)
    hwi = HWI(config, usb_port=args.port)

    joint_names = list(hwi.joints.keys())
    joint_ids = list(hwi.joints.values())

    print("Manual offset finder based on init_pos")
    print("This script does not move motors automatically.")
    print("")
    print("Move the robot by hand into the desired init pose.")
    print("Then press Enter to read raw motor positions and compute:")
    print("")
    print("offset = raw_position_at_desired_init_pose - init_pos")
    print("")

    print("Disabling torque on all motors...")
    hwi.io.disable_torque(joint_ids)
    time.sleep(0.5)

    input("Set the robot to the desired init pose by hand, then press Enter...")

    raw_positions = hwi.io.read_present_position(joint_ids)

    print("")
    print("Raw positions:")
    for name, raw in zip(joint_names, raw_positions):
        print(f"{name:16s}: {raw: .6f}")

    print("")
    print("Init positions:")
    for name in joint_names:
        print(f"{name:16s}: {hwi.init_pos[name]: .6f}")

    print("")
    print(f"Offset candidates for {HOME_DIR}/duck_config.json:")
    print("{")
    for name, raw in zip(joint_names, raw_positions):
        offset = raw - hwi.init_pos[name]
        print(f'  "{name}": {offset:.6f},')
    print("}")

    print("")
    print("Remove the trailing comma from the last line before pasting into JSON.")


if __name__ == "__main__":
    main()
