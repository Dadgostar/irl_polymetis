#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import subprocess
import atexit
import sys
import time
import signal

import hydra

from polymetis.utils.grpc_utils import check_server_exists
from polymetis.utils.data_dir import BUILD_DIR, which


log = logging.getLogger(__name__)


# config_path=None: the conf dir is provided on hydra's search path by
# hydra_plugins/polymetis_plugin (pkg://polymetis/conf), which is robust to how/where
# this script is installed. A relative config_path breaks once installed into bin/.
@hydra.main(config_path=None, config_name="launch_robot")
def main(cfg):
    log.info(f"Adding {BUILD_DIR} to $PATH")
    os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]

    # Check if another server is alive on address
    assert not check_server_exists(
        cfg.ip, cfg.port
    ), (
        "Port unavailable; possibly another server found on designated address. "
        "To prevent undefined behavior, start the service on a different port or "
        "kill stale servers with 'pkill -9 run_server'"
    )

    # Parse server address
    ip = str(cfg.ip)
    port = str(cfg.port)

    # Start server
    log.info(f"Starting server")
    server_exec_path = which(cfg.server_exec)
    server_cmd = [server_exec_path]
    server_cmd = server_cmd + ["-s", ip, "-p", port]

    if cfg.use_real_time:
        server_cmd += ["-r"]

    server_output = subprocess.Popen(
        server_cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setpgrp
    )
    pgid = os.getpgid(server_output.pid)

    # Kill process at the end
    def cleanup():
        log.info(f"Killing subprocess with pid {server_output.pid}, pgid {pgid}...")
        os.killpg(pgid, signal.SIGINT)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signal_number, stack_frame: cleanup())

    # Start client
    if cfg.robot_client:
        t0 = time.time()
        while not check_server_exists(cfg.ip, cfg.port):
            time.sleep(0.1)
            if time.time() - t0 > cfg.timeout:
                raise ConnectionError("Robot client: Unable to locate server.")

        log.info(f"Starting robot client...")
        # _recursive_=False: the client classes take their nested env_cfg/metadata_cfg
        # as DictConfigs and instantiate them themselves. Hydra >=1.1 instantiates
        # recursively by default, which would pre-build those and break the clients.
        client = hydra.utils.instantiate(cfg.robot_client, _recursive_=False)
        client.run()

    else:
        signal.pause()


if __name__ == "__main__":
    main()
