from pathlib import Path
from typing import Literal, TypeGuard, get_args

from vanilla_roll.util import has_module
from vanilla_roll.volume import Volume

FileFormat = Literal["mha"]

HAS_METAIMAGEIO = has_module("metaimageio")


def is_fileformat(format: str) -> TypeGuard[FileFormat]:
    return format in get_args(FileFormat)


def _get_filetype(path: str | Path, format: FileFormat | None) -> FileFormat | None:
    if format is not None:
        return format

    suffix = Path(path).suffix[1:]
    if is_fileformat(suffix):
        return suffix
    return None


def read(path: str | Path, /, format: FileFormat | None = None) -> Volume:
    match _get_filetype(path, format):
        case "mha":
            if not HAS_METAIMAGEIO:
                raise OSError("MetaImageIO is not installed")
            from vanilla_roll.io.mha import read_mha

            return read_mha(path)
        case _:
            raise ValueError(f"Unsupported file: {path}")


__all__ = ["read"]
