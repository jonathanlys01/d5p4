from typing import Any

is_master: bool
rank: int
local_rank: int
world_size: int
size: int

IdrTorchWarning: type[Warning]

def __getattr__(name: str) -> Any: ...
