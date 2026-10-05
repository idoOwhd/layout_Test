"""Bounded scratch growth during planning, never inside benchmark events.

This module has no torch dependency so failure/retry policy is CPU-testable.
Changing scratch capacity must not silently change backend/split-KV/layout.
"""
from __future__ import annotations
import re

INITIAL_BYTES=128*1024*1024
MAX_BYTES=2*1024*1024*1024
OVERFLOW=re.compile(
    r"Buffer overflow when allocating memory for (?P<allocation>\w+) "
    r"with size (?P<requested>\d+) and alignment (?P<alignment>\d+), "
    r"but only (?P<remaining>\d+) bytes available in AlignedAllocator")


class WorkspaceLimitError(RuntimeError):
    pass


def growth_request(error,current_bytes,max_bytes):
    match=OVERFLOW.search(str(error))
    if match is None:return None
    info={k:int(v) if k!="allocation" else v for k,v in match.groupdict().items()}
    # Retry only float attention scratch overflows. Integer metadata/other
    # allocators cannot be repaired by changing the float workspace.
    if not info["allocation"].endswith(("tmp_v","tmp_s")):return None
    if not (0<=info["remaining"]<=current_bytes) or info["alignment"]<1:
        raise ValueError("inconsistent FlashInfer workspace error")
    needed=current_bytes-info["remaining"]+info["requested"]+info["alignment"]-1
    target=1 << (max(2*current_bytes,needed)-1).bit_length()
    if needed>max_bytes:
        raise WorkspaceLimitError(f"FlashInfer needs at least {needed} scratch bytes; "
                                  f"configured workspace limit is {max_bytes}. Original error: {error}")
    target=min(target,max_bytes)
    if target<=current_bytes:raise WorkspaceLimitError("FlashInfer workspace growth exhausted")
    return {**info,"minimum_required_bytes":needed,"next_bytes":target}


def plan_with_workspace(factory,plan,allocate,initial_buffer,initial_bytes=INITIAL_BYTES,max_bytes=MAX_BYTES):
    """Return the planned wrapper and an auditable allocation history.

    factory(buffer) and plan(wrapper) must keep all algorithm parameters fixed.
    A shared initial buffer is retained for cases that already worked, keeping
    the previous 128-MiB setup unchanged. Failed plans are rebuilt from scratch.
    """
    if not 0<initial_bytes<=max_bytes:raise ValueError("invalid workspace size limits")
    size=initial_bytes;buffer=initial_buffer;attempts=[]
    while True:
        wrapper=factory(buffer)
        try:
            plan(wrapper)
        except RuntimeError as error:
            growth=growth_request(error,size,max_bytes)
            if growth is None:raise
            attempts.append({"bytes":size,"status":"workspace_overflow",**growth})
            wrapper=None
            buffer=None
            # A new wrapper owns the new scratch; never mutate an existing
            # successfully planned wrapper's allocation behind its back.
            size=growth["next_bytes"]
        else:
            attempts.append({"bytes":size,"status":"planned"})
            return wrapper,{"initial_bytes":initial_bytes,"final_bytes":size,
                "max_bytes":max_bytes,"attempts":attempts,"growth_outside_timed_region":True,
                "backend_or_split_policy_changed":False}
        buffer=allocate(size)
