"""Replace or remove staged artifacts, with rollback on I/O errors."""

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from shutil import copy2
from stat import S_IMODE
from tempfile import TemporaryDirectory


def publish_files(changes: Mapping[Path, Path | None]) -> None:
    with ExitStack() as stack:
        backup_dirs = {}
        backups = {}
        for target, staged in changes.items():
            backup = None
            if target.exists():
                if target.parent not in backup_dirs:
                    backup_dirs[target.parent] = Path(
                        stack.enter_context(TemporaryDirectory(dir=target.parent))
                    )
                backup = backup_dirs[target.parent] / target.name
                copy2(target, backup)
                if staged is not None:
                    staged.chmod(S_IMODE(backup.stat().st_mode))
            backups[target] = backup

        published = []
        try:
            for target, staged in changes.items():
                if staged is None:
                    target.unlink(missing_ok=True)
                else:
                    staged.replace(target)
                published.append(target)
        except OSError:
            for target in reversed(published):
                backup = backups[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    backup.replace(target)
            raise
