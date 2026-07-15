# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from concurrent import futures
import json

import grpc

import polymetis_pb2
import polymetis_pb2_grpc


class PolymetisGripperServer(polymetis_pb2_grpc.GripperServerServicer):
    """gRPC server that exposes a gripper controls to the user client
    Gripper connects as a second client
    """

    def __init__(self):
        self.gripper_state_cache = polymetis_pb2.GripperState()
        self.gripper_cmd_cache = polymetis_pb2.GripperCommand()
        self.metadata = None

    def InitRobotClient(self, request, context):
        self.metadata = request
        return polymetis_pb2.Empty()

    def GetRobotClientMetadata(self, request, context):
        return self.metadata

    def ControlUpdate(self, request, context):
        self.gripper_state_cache = request
        return self.gripper_cmd_cache

    def GetState(self, request, context):
        return self.gripper_state_cache

    def Goto(self, request, context):
        self.gripper_cmd_cache = request
        return polymetis_pb2.Empty()


class GripperServerLauncher:
    def __init__(self, ip="localhost", port="50052"):
        self.address = f"{ip}:{port}"
        # so_reuseport=0: gRPC's default SO_REUSEPORT lets a SECOND server bind the
        # same port, silently load-balancing client connections between old and new
        # processes -- commands then intermittently land in a cache no hand client
        # polls (grasps work, opens vanish, width reads 0). Disable it so a stale
        # server makes a relaunch FAIL LOUDLY here instead.
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2),
            options=[("grpc.so_reuseport", 0)],
        )

        polymetis_pb2_grpc.add_GripperServerServicer_to_server(
            PolymetisGripperServer(), self.server
        )
        if self.server.add_insecure_port(self.address) == 0:
            raise SystemExit(
                f"[gripper-server] could not bind {self.address} -- another gripper "
                f"server is already running on this port. Kill it first "
                f"(pkill -f 'launch_gripper.*{port}') and relaunch."
            )

    def run(self):
        self.server.start()
        print(f"Gripper server running at {self.address}.")
        self.server.wait_for_termination()
