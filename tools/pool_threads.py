"""CPU budget for dataset workers; set before starting a forkserver pool."""
import os


def igl_threads(workers: int) -> str:
    """Respect the caller's IGL_NUM_THREADS; otherwise split the CPU budget.

    CGAL/libigl thread loops do not infer a container's CPU quota. This helper
    belongs to the CLI, not the library: importing volumetric_iou never changes
    the calling process's thread settings.
    """
    n_cpu = len(os.sched_getaffinity(0))
    try:
        with open("/sys/fs/cgroup/cpu.max") as stream:
            quota, period = stream.read().split()
        if quota != "max":
            n_cpu = min(n_cpu, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    os.environ.setdefault("IGL_NUM_THREADS", str(max(1, min(8, n_cpu // max(1, workers)))))
    return os.environ["IGL_NUM_THREADS"]
