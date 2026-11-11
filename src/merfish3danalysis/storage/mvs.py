"""Lazy TensorStore-backed array view for multiview-stitcher."""

import json
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

import numpy as np
import zarr
from tensorstore import TensorStore
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype


class _TensorStoreChunkView(Store):
    """Expose inner chunks because MVS cannot expand sharded Zarr arrays yet.

    Parameters
    ----------
    source : tensorstore.TensorStore
        Lazily opened source image array.
    metadata : dict[str, Any]
        Zarr v3 source array metadata, including shape and fill value.
    chunks : Sequence[int] or None, default None
        Requested view chunk shape in array-axis order, or None to use the
        source's inner read chunks.
    """

    supports_writes = False
    supports_deletes = False
    supports_listing = False

    def __init__(
        self,
        source: TensorStore,
        metadata: dict[str, Any],
        chunks: Sequence[int] | None = None,
    ) -> None:
        """Initialize a read-only TensorStore chunk view.

        Parameters
        ----------
        source : tensorstore.TensorStore
            Lazily opened source image array.
        metadata : dict[str, Any]
            Zarr v3 source array metadata, including shape and fill value.
        chunks : Sequence[int] or None, default None
            Requested view chunk shape in array-axis order, or None to use the
            source's inner read chunks.
        """
        super().__init__(read_only=True)
        self.source = source
        self.shape = tuple(metadata["shape"])
        self.chunks = tuple(chunks or source.chunk_layout.read_chunk.shape)
        self.dtype = np.dtype(source.dtype.numpy_dtype)
        self.fill_value = metadata["fill_value"]
        metadata = dict(metadata)
        metadata["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": self.chunks},
        }
        metadata["chunk_key_encoding"] = {
            "name": "default",
            "configuration": {"separator": "/"},
        }
        metadata["codecs"] = [{"name": "bytes", "configuration": {"endian": "little"}}]
        self.metadata = json.dumps(metadata).encode()

    def __eq__(self, other: object) -> bool:
        """Compare views by object identity.

        Parameters
        ----------
        other : object
            Store to compare with this view by object identity.
        """
        return self is other

    def __hash__(self) -> int:
        """Return an identity-based hash for the chunk view."""
        return id(self)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Read a metadata value or lazily encode one TensorStore chunk.

        Parameters
        ----------
        key : str
            Zarr metadata or chunk key to read from the view.
        prototype : zarr.core.buffer.BufferPrototype
            Buffer implementation used for the returned bytes.
        byte_range : zarr.abc.store.ByteRequest or None, default None
            Requested byte range within the value, or None for the entire value.
        """
        if key == "zarr.json":
            data = self.metadata
        elif key.startswith("c/"):
            try:
                coords = tuple(int(part) for part in key[2:].split("/"))
            except ValueError:
                return None
            if len(coords) != len(self.shape) or any(
                index < 0 or index * chunk >= size
                for index, chunk, size in zip(
                    coords, self.chunks, self.shape, strict=True
                )
            ):
                return None
            slices = tuple(
                slice(index * chunk, min((index + 1) * chunk, size))
                for index, chunk, size in zip(
                    coords, self.chunks, self.shape, strict=True
                )
            )
            pixels = await self.source[slices].read()
            if pixels.shape != self.chunks:
                padded = np.full(
                    self.chunks, self.fill_value, dtype=self.dtype.newbyteorder("<")
                )
                padded[tuple(slice(0, size) for size in pixels.shape)] = pixels
                pixels = padded
            data = pixels.astype(self.dtype.newbyteorder("<"), copy=False).tobytes(
                order="C"
            )
        else:
            return None
        if isinstance(byte_range, RangeByteRequest):
            data = data[byte_range.start : byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = data[byte_range.offset :]
        elif isinstance(byte_range, SuffixByteRequest):
            data = data[-byte_range.suffix :]
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Read requested keys and byte ranges in order.

        Parameters
        ----------
        prototype : zarr.core.buffer.BufferPrototype
            Buffer implementation used for each returned value.
        key_ranges : Iterable[tuple[str, ByteRequest or None]]
            Metadata or chunk keys paired with their requested byte ranges.
        """
        return [
            await self.get(key, prototype, byte_range) for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        """Check whether a metadata value or chunk is available.

        Parameters
        ----------
        key : str
            Zarr metadata or chunk key whose presence is checked.
        """
        return await self.get(key, default_buffer_prototype()) is not None

    async def set(self, key: str, value: Buffer) -> None:
        """Reject writes to the MVS input view.

        Parameters
        ----------
        key : str
            Destination key supplied by the Zarr Store protocol.
        value : zarr.core.buffer.Buffer
            Bytes supplied for writing. This read-only view rejects the write.
        """
        raise NotImplementedError("MVS input view is read-only")

    async def delete(self, key: str) -> None:
        """Reject deletion from the MVS input view.

        Parameters
        ----------
        key : str
            Key supplied for deletion. This read-only view rejects the deletion.
        """
        raise NotImplementedError("MVS input view is read-only")

    async def list(self) -> AsyncIterator[str]:
        """Reject key listing, which the MVS input view does not support."""
        raise NotImplementedError("MVS input view does not support listing")
        yield

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """Reject key-prefix listing for the MVS input view.

        Parameters
        ----------
        prefix : str
            Key prefix supplied for listing, which this view does not support.
        """
        return self.list()

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Reject directory listing for the MVS input view.

        Parameters
        ----------
        prefix : str
            Directory prefix supplied for listing, which this view does not support.
        """
        return self.list()


def mvs_array(array: Any, chunks: Sequence[int] | None = None) -> zarr.Array:
    """Adapt sharded yaozarrs arrays for MVS singleton-axis and channel views.

    Parameters
    ----------
    array : Any
        Open yaozarrs array.
    chunks : Sequence[int] or None
        Requested view chunk shape, or None to use the native layout.

    Returns
    -------
    zarr.Array
        Lazy MVS-compatible view of the array.
    """
    metadata = array.metadata.model_dump(mode="json")
    if chunks is None and not any(
        codec["name"] == "sharding_indexed" for codec in metadata["codecs"]
    ):
        return array.to_zarr_python()
    return zarr.open_array(
        store=_TensorStoreChunkView(array.to_tensorstore(), metadata, chunks), mode="r"
    )


def shared_mvs_chunks(arrays: Sequence[zarr.Array]) -> tuple[int, ...] | None:
    """Match mixed channel views so MVS can concatenate them without loading them.

    Parameters
    ----------
    arrays : Sequence[zarr.Array]
        Native or TensorStore-backed image arrays to combine.

    Returns
    -------
    tuple[int, ...] or None
        Minimum chunk size along each axis when a view is needed; otherwise None.
    """
    if not any(isinstance(array.store, _TensorStoreChunkView) for array in arrays):
        return None
    return tuple(
        min(chunks) for chunks in zip(*(array.chunks for array in arrays), strict=True)
    )
