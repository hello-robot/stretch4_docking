import zmq
import time
import struct
import numpy as np
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


def _as_origin_xy(origin) -> tuple[float, float]:
    values = np.asarray(origin, dtype=np.float64).ravel()
    if values.size == 1:
        return float(values[0]), float(values[0])
    if values.size == 2:
        return float(values[0]), float(values[1])
    raise ValueError(f"origin must be a scalar or an (x, y) pair, got {values.size} elements")


class Mppi:

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

    def request(self, costmap: np.ndarray, origin, resolution: float, goal, goal_valid: bool = True, reset: bool = False, elapsed_s: float = 0.0, state=(0.0, 0.0, 0.0)) -> MppiReply | None:
        if self.socket is None:
            self.connect()

        cells = np.ascontiguousarray(costmap, dtype=np.float32)
        if cells.ndim != 2:
            raise ValueError(f"costmap must be a 2D grid, got {cells.ndim} dimensions")
        origin_x, origin_y = _as_origin_xy(origin)

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
        if reconnect:
            self.connect()
        return None
