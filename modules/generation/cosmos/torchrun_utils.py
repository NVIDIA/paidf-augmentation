# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for torchrun-based local executors."""

import random
import socket
from typing import Tuple


def reserve_free_port() -> Tuple[int, socket.socket]:
    """Reserve a free port in the IANA dynamic/private range (49152-65535).

    Returns the port and the bound socket so the caller can keep the socket
    alive (holding the reservation) until just before the child process binds.
    ``SO_REUSEADDR`` is set so closing the socket releases the port cleanly
    for an immediate rebind by the child.

    There is a narrow TOCTOU window between the caller closing the socket
    and the child binding; callers should close the socket as late as possible
    to minimize it.
    """
    ports = list(range(49152, 65536))
    random.shuffle(ports)
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
            return port, sock
        except OSError:
            sock.close()
    raise RuntimeError("No free port found in IANA dynamic range 49152-65535")
