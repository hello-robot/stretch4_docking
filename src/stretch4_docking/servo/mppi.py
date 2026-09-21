import zmq
import time
import struct
import socket
import numpy as np
from loguru import logger
from dataclasses import dataclass

REQUEST_HEADER = struct.Struct('<iifffffffifffii')
REPLY_HEADER = struct.Struct('<ifffffi')


@dataclass(frozen=True)
class MppiConfig:
    host: str = '192.168.1.101'
    port: int = 5556
    recv_timeout_ms: int = 1000
    send_timeout_ms: int = 200

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass(frozen=True)
class MppiReply:
    vx: float
    vy: float
    wz: float
    stiffness: float
    engaged: bool
    service_ms: float
    wire_ms: float

    @property
    def cmd(self) -> tuple[float, float, float]:
        return self.vx, self.vy, self.wz


class Mppi:

    @staticmethod
    def is_online(config: MppiConfig | None = None) -> None:
        """Checks if port on the Jetson accepts a connection."""
        config = config if config is not None else MppiConfig()
        try:
            with socket.create_connection((config.host, config.port), timeout=2):
                return True
        except OSError:
            return False

    def __init__(self, config: MppiConfig | None = None):
        self.config = config if config is not None else MppiConfig()
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self.frame_counter = 0
        self.needs_reset = True
        self.last_error: str | None = None
        self.connect()

    @property
    def endpoint(self) -> str:
        return self.config.endpoint

    def connect(self) -> None:
        self.close()
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, int(self.config.recv_timeout_ms))
        socket.setsockopt(zmq.SNDTIMEO, int(self.config.send_timeout_ms))
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.config.endpoint)
        self.socket = socket

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def __enter__(self) -> "Mppi":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        """Force the next request() to carry the reset flag. Call when starting or restarting a servo attempt."""
        self.needs_reset = True

    def step(self, err_x: float, err_y: float, err_theta: float, costmap: np.ndarray, origin_x: float, origin_y: float, resolution: float,
              goal_valid: bool = True, reset: bool = False, elapsed_s: float = 0.0,
              state=(0.0, 0.0, 0.0)) -> tuple[float, float, float, float | None]:
        reply = self.request(costmap, origin_x, origin_y, resolution, goal=(err_x, err_y, err_theta),
                              goal_valid=goal_valid, reset=reset, elapsed_s=elapsed_s, state=state)
        if reply is None:
            return 0.0, 0.0, 0.0, 1.0
        return reply.vx, reply.vy, reply.wz, reply.stiffness

    def request(self, costmap: np.ndarray, origin_x, origin_y, resolution: float, goal, goal_valid: bool = True, reset: bool = False, elapsed_s: float = 0.0, state=(0.0, 0.0, 0.0)) -> MppiReply | None:
        if self.socket is None:
            self.connect()

        cells = np.ascontiguousarray(costmap, dtype=np.float32)
        if cells.ndim != 2:
            raise ValueError(f"costmap must be a 2D grid, got {cells.ndim} dimensions")

        self.frame_counter += 1
        header = REQUEST_HEADER.pack(
            self.frame_counter, int(reset or self.needs_reset), float(elapsed_s),
            float(state[0]), float(state[1]), float(state[2]),
            float(goal[0]), float(goal[1]), float(goal[2]),
            int(goal_valid),
            origin_x, origin_y, float(resolution),
            cells.shape[0], cells.shape[1])

        try:
            rtt_start = time.perf_counter()
            self.socket.send_multipart([header, cells.tobytes()])
            payload = self.socket.recv()
            rtt_ms = (time.perf_counter() - rtt_start) * 1000.0
        except zmq.error.Again:
            return self._fail(f"plant did not respond at {self.config.endpoint}", reconnect=True)

        if len(payload) != REPLY_HEADER.size:
            return self._fail(f"malformed plant reply ({len(payload)} bytes, "
                              f"expected {REPLY_HEADER.size})")

        _frame, vx, vy, wz, stiffness, service_ms, engaged = REPLY_HEADER.unpack(payload)

        self.needs_reset = False
        self.last_error = None
        return MppiReply(
            vx=vx, vy=vy, wz=wz,
            stiffness=stiffness,
            engaged=bool(engaged),
            service_ms=service_ms,
            # The plant reports how long it spent on this request, so whatever
            # is left of the round trip is serialisation plus network. The two
            # are worth separating because the plant usually runs on another
            # machine, where a slow cycle is not necessarily a slow GPU.
            wire_ms=max(rtt_ms - service_ms, 0.0),
        )

    def _fail(self, message: str, reconnect: bool = False) -> None:
        self.last_error = message
        self.needs_reset = True
        logger.error(f"{self.last_error=} {self.need_reset=}")
        if reconnect:
            self.connect()
        return None
