import argparse
import time

import rustypot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Motor ID to monitor")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Motor controller port")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--disable-torque", action="store_true")
    args = parser.parse_args()

    io = rustypot.feetech(args.port, 1000000)

    if args.disable_torque:
        print(f"Disabling torque on motor ID {args.id}")
        io.disable_torque([args.id])
        time.sleep(0.2)

    print(f"Watching motor ID {args.id}")
    print("Move the joint by hand. Press Ctrl+C to stop.")
    print("")

    try:
        while True:
            try:
                pos = io.read_present_position([args.id])[0]
                print(f"motor {args.id}: {pos:.3f} rad")
            except Exception as e:
                print(f"read error: {e}")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
