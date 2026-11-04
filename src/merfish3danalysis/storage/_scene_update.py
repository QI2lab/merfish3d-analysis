"""Fill yaozarrs' gap for atomic updates to an existing local Scene."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from filelock import FileLock
from yaozarrs import open_group, v06
from yaozarrs.write._common._groups import _create_zarr3_group

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike


def _local_scene_path(dest: str | PathLike) -> Path:
    """Restrict atomic publication to local POSIX paths and canonicalize aliases."""
    if os.name != "posix" or "://" in str(dest):
        raise ValueError("Scene updates require a local POSIX filesystem.")
    return Path(dest).resolve(strict=True)


def _scene_lock(path: Path) -> FileLock:
    """Serialize independent workers updating the same Scene."""
    return FileLock(str(path.parent / f".{path.name}.metadata.lock"))


def ensure_scene_group(dest: str | PathLike, relative_path: str) -> Path:
    """Create a parent group that yaozarrs' standalone image writer cannot create.

    Parameters
    ----------
    dest : str or PathLike
        Local OME-NGFF 0.6 Scene directory.
    relative_path : str
        Relative group path beneath the Scene. Parent groups are created as
        needed; existing groups are preserved and symbolic links are rejected.

    Returns
    -------
    Path
        Path to the requested group within the Scene.
    """
    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in (".", "..") for part in relative_path.split("/"))
    ):
        raise ValueError("A relative Scene group path without '.' or '..' is required.")
    root = _local_scene_path(dest)
    with _scene_lock(root):
        if not isinstance(open_group(root).ome_metadata(strict=True), v06.Scene):
            raise ValueError("Expected an OME-NGFF 0.6 Scene.")
        group = root
        for part in relative.parts:
            group = group / part
            if group.is_symlink():
                raise ValueError("Scene group paths cannot traverse symbolic links.")
            if group.exists():
                open_group(group)
            else:
                # yaozarrs has no public API for adding a plain group to an
                # existing Scene; its own writer keeps group metadata upstream.
                _create_zarr3_group(group)
        return group


def update_scene(
    dest: str | PathLike,
    update: Callable[[v06.Scene, dict[str, Any]], tuple[v06.Scene, dict[str, Any]]],
) -> None:
    """Atomically update metadata because yaozarrs cannot resume a Scene writer.

    Parameters
    ----------
    dest : str or PathLike
        Local Scene directory. Different symlink paths to the same Scene use the
        same interprocess lock.
    update : callable
        Called while holding the lock with a freshly opened Scene and its extra
        root attributes, excluding the reserved ``ome`` key. Return the desired
        Scene and extra attributes. Raising an exception leaves the Scene intact.

    Notes
    -----
    Validation runs against a staged root linked to existing child groups. Child
    groups referenced by a committed Scene must be immutable while readers use
    them. This operation does not create arrays or write image data. It supports
    local POSIX filesystems only; the caller must ensure that file locking,
    symlinks and atomic rename have the expected semantics on that filesystem.
    """
    path = _local_scene_path(dest)
    with _scene_lock(path):
        group = open_group(path)
        current = group.ome_metadata(strict=True)
        if not isinstance(current, v06.Scene):
            raise ValueError("Expected an OME-NGFF 0.6 Scene.")
        attributes = dict(group.attrs)
        attributes.pop("ome", None)
        scene, attributes = update(current, attributes)
        if not isinstance(scene, v06.Scene) or "ome" in attributes:
            raise ValueError("Update must return a Scene and non-OME attributes.")
        scene = v06.Scene.model_validate(scene.model_dump(mode="python"), strict=True)
        attributes = dict(attributes)
        attributes["ome"] = scene.model_dump(mode="json")
        metadata = group.metadata.model_copy(update={"attributes": attributes})

        with TemporaryDirectory(
            prefix=f".{path.name}.validate-", dir=path.parent
        ) as temp:
            candidate = Path(temp)
            for child in path.iterdir():
                if child.name != "zarr.json":
                    (candidate / child.name).symlink_to(child.resolve())
            candidate_metadata = candidate / "zarr.json"
            with candidate_metadata.open("w", encoding="utf-8") as stream:
                stream.write(metadata.model_dump_json(exclude_none=True))
                stream.flush()
                os.fsync(stream.fileno())
            open_group(candidate).validate(strict=True)
            os.replace(candidate_metadata, path / "zarr.json")
            directory = os.open(path, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
