"""Native thread budget for dataset workers, before starting a forkserver pool."""
import os


def igl_threads(workers: int) -> str:
    """Respect IGL_NUM_THREADS; otherwise keep the package's one-thread default.

    CGAL/libigl thread loops do not infer a container's CPU quota. This helper
    belongs to the CLI, not the library: importing volumetric_iou never changes
    the calling process's thread settings.
    """
    os.environ.setdefault("IGL_NUM_THREADS", "1")
    return os.environ["IGL_NUM_THREADS"]
