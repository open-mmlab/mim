# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) https://github.com/pypa/pip
# Modified from https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/progress_bars.py  # noqa: E501

from typing import Callable, Generator, Iterable, Iterator

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

DownloadProgressRenderer = Callable[[Iterable[bytes]], Iterator[bytes]]


def rich_progress_bar(
    iterable: Iterable[bytes],
    size: int,
) -> Generator[bytes, None, None]:
    columns = (
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        DownloadColumn(binary_units=1024),
        TransferSpeedColumn(),
        TextColumn('eta'),
        TimeRemainingColumn(),
    )

    progress = Progress(*columns, refresh_per_second=30)
    task_id = progress.add_task('downloading', total=size)
    with progress:
        for chunk in iterable:
            yield chunk
            progress.update(task_id, advance=len(chunk))
