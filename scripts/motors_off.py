import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_bdx_runtime"))

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(config, usb_port=args.port)

    ids = list(hwi.joints.values())
    print("Disabling torque:", ids)
    hwi.io.disable_torque(ids)
    print("done")


if __name__ == "__main__":
    main()
