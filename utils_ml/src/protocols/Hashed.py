from typing_extensions import Protocol


class Hashed(Protocol):
    def __hash__(self) -> int: ...
