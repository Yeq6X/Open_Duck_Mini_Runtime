import argparse
import select
import sys
import time

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True, help="Joint name, e.g. right_ankle")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    parser.add_argument("--disable-all", action="store_true")
    args = parser.parse_args()

    config = DuckConfig(config_json_path=None, ignore_default=True)
    hwi = HWI(config, usb_port=args.port)

    if args.joint not in hwi.joints:
        print(f"Unknown joint: {args.joint}")
        print("")
        print("Available joints:")
        for name, motor_id in hwi.joints.items():
            print(f"  {name:16s} ID {motor_id}")
        sys.exit(1)

    joint_id = hwi.joints[args.joint]
    init_pos = hwi.init_pos[args.joint]

    if args.disable_all:
        print("Disabling torque on all motors...")
        hwi.io.disable_torque(list(hwi.joints.values()))
    else:
        print(f"Disabling torque on {args.joint} / ID {joint_id}...")
        hwi.io.disable_torque([joint_id])

    time.sleep(0.2)

    print("")
    print(f"Watching joint: {args.joint}")
    print(f"Motor ID      : {joint_id}")
    print(f"init_pos      : {init_pos:.6f} rad")
    print("")
    print("Move this joint by hand to the desired init-pose position.")
    print("Press Enter to print the current offset candidate.")
    print("Press Ctrl+C to stop.")
    print("")

    last_line_len = 0
    latest_offset = None

    try:
        while True:
            raw = hwi.io.read_present_position([joint_id])[0]
            latest_offset = raw - init_pos

            line = (
                f"{args.joint} | "
                f"raw={raw:+.6f} rad | "
                f"init={init_pos:+.6f} rad | "
                f"offset_candidate={latest_offset:+.6f} rad"
            )

            padding = " " * max(0, last_line_len - len(line))
            print("\r" + line + padding, end="", flush=True)
            last_line_len = len(line)

            if select.select([sys.stdin], [], [], args.interval)[0]:
                sys.stdin.readline()
                break

    except KeyboardInterrupt:
        print("\nStopped.")
        return

    print("")
    print("")
    print("Confirmed offset candidate:")
    print(f'  "{args.joint}": {latest_offset:.6f},')
    print("")
    print("Put this value in duck_config.json joints_offsets.")

    if args.disable_all:
        hwi.io.disable_torque(list(hwi.joints.values()))
    else:
        hwi.io.disable_torque([joint_id])


if __name__ == "__main__":
    main()
