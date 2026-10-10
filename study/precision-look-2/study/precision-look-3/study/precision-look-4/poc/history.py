"""Bounded physical-time contexts, copied independently for every candidate."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


def _immutable_numpy(value, dtype):
    array = np.asarray(value,dtype=dtype)
    return np.frombuffer(array.tobytes(),dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class HistoryToken:
    start_ns: int
    state: np.ndarray | torch.Tensor
    mean_command: np.ndarray
    duration_ns: int
    provenance: str
    packet_id: str

    def __post_init__(self):
        if type(self.start_ns) is not int or type(self.duration_ns) is not int or self.start_ns < 0 or self.duration_ns <= 0:
            raise ValueError('History requires nonnegative integer nanoseconds and positive duration')
        if self.provenance not in {'observed','imagined'}:
            raise ValueError('Unknown history provenance')
        if not isinstance(self.packet_id,str) or not self.packet_id:
            raise ValueError('History packet identity is required')
        if isinstance(self.state,torch.Tensor):
            state = self.state.clone()
            finite = bool(torch.isfinite(state).all())
        else:
            state = _immutable_numpy(self.state,np.float64)
            finite = np.isfinite(state).all()
        command = _immutable_numpy(self.mean_command,np.float64)
        if state.shape not in {(4,),(6,)} or not finite or command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command)>2):
            raise ValueError('History state/physical mean command has invalid shape or values')
        object.__setattr__(self,'state',state)
        object.__setattr__(self,'mean_command',command)

    def copy(self):
        # The NumPy arrays were copied into immutable bytes at ingestion. Sharing
        # those bytes is safe; Torch payloads remain mutable and need a clone.
        if isinstance(self.state,np.ndarray):
            return self
        return HistoryToken(self.start_ns,self.state,self.mean_command,self.duration_ns,self.provenance,self.packet_id)


@dataclass(frozen=True)
class HistoryBatch:
    tokens: tuple[tuple[HistoryToken,...],...]
    query_time_ns: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.query_time_ns)
        if times.dtype != np.int64 or times.shape != (len(self.tokens),) or np.any(times < 0):
            raise ValueError('One nonnegative integer nanosecond query time per candidate is required')
        copies = []
        for sequence in self.tokens:
            if len({token.packet_id for token in sequence}) != len(sequence):
                raise ValueError('Duplicate packet identity in a history branch')
            if any(a.start_ns >= b.start_ns for a,b in zip(sequence,sequence[1:])):
                raise ValueError('History token starts must increase in physical time')
            copies.append(tuple(token.copy() for token in sequence))
        object.__setattr__(self,'tokens',tuple(copies))
        object.__setattr__(self,'query_time_ns',_immutable_numpy(times,np.int64))

    def eligible(self):
        return HistoryBatch(tuple(tuple(token for token in sequence
                 if int(time)-320_000_000 <= token.start_ns < int(time)
                 and token.start_ns+token.duration_ns <= int(time))
                 for sequence,time in zip(self.tokens,self.query_time_ns)),self.query_time_ns)

    def fork(self):
        return HistoryBatch(self.tokens,self.query_time_ns)

    def at_time(self, query_time_ns: np.ndarray):
        return HistoryBatch(self.tokens,query_time_ns)

    def append_completed(self, token: HistoryToken | tuple[HistoryToken,...]):
        additions = (token,) if isinstance(token,HistoryToken) else token
        if len(additions) != len(self.tokens):
            raise ValueError('Append one completed token per candidate')
        if any(value.start_ns+value.duration_ns > int(time) for value,time in zip(additions,self.query_time_ns)):
            raise ValueError('Only completed transitions may enter history')
        return HistoryBatch(tuple((*sequence,value) for sequence,value in zip(self.tokens,additions)),self.query_time_ns)


def mean_command(segments: tuple, start_ns: int, end_ns: int) -> np.ndarray:
    if type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns:
        raise ValueError('A positive completed observation interval is required')
    total = np.zeros(2,dtype=np.float64)
    cursor = start_ns
    for start,end,command in segments:
        values = np.asarray(command,dtype=np.float64)
        if start != cursor or end <= start or end > end_ns or values.shape != (2,) or not np.isfinite(values).all() or np.any(np.abs(values)>2):
            raise ValueError('Command segments must be contiguous and cover the interval exactly')
        total += (end-start)*values
        cursor = end
    if cursor != end_ns:
        raise ValueError('Command segments must cover the entire completed interval')
    return total/(end_ns-start_ns)
