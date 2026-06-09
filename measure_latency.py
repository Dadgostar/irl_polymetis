"""Measure prev_controller_latency_ms from a live polymetis server.

Usage (sim or real robot, server must already be running):
    pixi run python measure_latency.py [--ip localhost] [--duration 10]
"""
import argparse
import time

import numpy as np
from polymetis import RobotInterface


def measure(ip: str = "localhost", port: int = 50051, duration: float = 10.0) -> None:
    robot = RobotInterface(ip_address=ip, port=port)

    print(f"Connected to {ip}:{port}. Going home...")
    robot.go_home()

    print(f"Starting joint impedance for {duration}s...")
    robot.start_joint_impedance()
    time.sleep(duration)

    log = robot.terminate_current_policy()
    print(f"Collected {len(log)} states.")

    lat = np.array([s.prev_controller_latency_ms for s in log])
    lat = lat[lat > 0]  # first cycle is unset (0)

    if len(lat) == 0:
        print("No latency data found (all zeros). Is the server reporting latency?")
        return

    print(f"\n--- Controller latency over {len(lat)} cycles ---")
    print(f"  mean:   {lat.mean():.3f} ms")
    print(f"  std:    {lat.std():.3f} ms")
    print(f"  min:    {lat.min():.3f} ms")
    print(f"  max:    {lat.max():.3f} ms")
    for p in (50, 95, 99, 99.9):
        print(f"  p{p:<5}: {np.percentile(lat, p):.3f} ms")

    # Flag cycles that exceeded 1ms budget (1kHz server)
    over = (lat > 1.0).sum()
    print(f"\n  Cycles > 1.0 ms: {over} ({100 * over / len(lat):.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="How many seconds to run joint impedance (default: 10)")
    args = parser.parse_args()
    measure(args.ip, args.port, args.duration)
