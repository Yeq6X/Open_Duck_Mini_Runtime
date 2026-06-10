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
    parser.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    parser.add_argument("--hold", type=float, default=5.0, help="Hold time in seconds")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=args.duck_config_path)
    hwi = HWI(config, usb_port=args.port)

    try:
        print("Turning on and moving to init_pos...")
        hwi.turn_on()
        print(f"Holding init_pos for {args.hold} seconds.")
        time.sleep(args.hold)
    finally:
        print("Turning off torque.")
        hwi.turn_off()


if __name__ == "__main__":
    main()
