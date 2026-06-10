import argparse
import time

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=None, ignore_default=True)
    hwi = HWI(config, usb_port=args.port)

    joint_ids = list(hwi.joints.values())

    print("Manual soft offset finder")
    print("This script does not move motors automatically.")
    print("It disables torque and reads raw motor positions.")
    print("Move each joint by hand to the desired zero pose, then press Enter.")
    print("Press Ctrl+C to stop.")
    print("")

    print("Disabling torque on all motors...")
    hwi.io.disable_torque(joint_ids)
    time.sleep(0.5)

    offsets = {}

    try:
        for joint_name, joint_id in hwi.joints.items():
            print("")
            print(f"=== {joint_name} / ID {joint_id} ===")

            res = input("Adjust this joint? [Y/s] ").lower()
            if res == "s":
                print("Skipped.")
                continue

            hwi.io.disable_torque([joint_id])
            time.sleep(0.2)

            input("Move this joint by hand to the desired zero position, then press Enter...")

            raw_pos = hwi.io.read_present_position([joint_id])[0]
            offsets[joint_name] = raw_pos

            print(f"Offset candidate for {joint_name}: {raw_pos:.6f}")
            print("")
            print("Current offsets:")
            for name, value in offsets.items():
                print(f'  "{name}": {value:.6f},')

        print("")
        print("Done.")
        print("Copy these values into duck_config.json joints_offsets:")
        print("{")
        for name, value in offsets.items():
            print(f'  "{name}": {value:.6f},')
        print("}")

    finally:
        print("Disabling torque on all motors...")
        hwi.io.disable_torque(joint_ids)


if __name__ == "__main__":
    main()
