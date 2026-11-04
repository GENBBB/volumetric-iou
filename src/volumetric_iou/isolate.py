"""
Run one call in a child process with a time limit.

A hang inside native code (manifold, CGAL) cannot be interrupted from the
same process, hence a child per call: the call and its result go through a
socket as pickles, the caller waits poll(timeout), then terminates/kills the
child; the cause of death comes from the child's exit status.

The child is not forked from the caller but from a server process (one per
caller process, started at the first call): a fork copies the thread pools a
library has started in the caller (oneTBB in the PyPI manifold3d), and a
child forked while their workers are busy may wait for them forever.  The
server only imports the package and the modules it imports lazily (no
geometry, so no pool of its own) and forks a child per call.  Arguments and
the result therefore must pickle (a call that does not raises in the caller);
a function of the caller's main script (when `fn` itself is one) is taken
from its file, which the child imports as `__mp_main__` (guard the script's
own work with `if __name__ == "__main__":`).

The server is started with the caller's environment and sys.path as they
are at that moment, and started again if it has died or the package is now
imported from another file; a child also gets the caller's sys.path of the
call.  The server ignores SIGINT and exits, killing its children, when the
caller's end of its socket closes -- at the caller's exit or death.  A child
dies with the server (PR_SET_PDEATHSIG) and checks the server is still its
parent.  A server that cannot start raises (RuntimeError, or the OSError of
starting the interpreter).  Linux only.

`run_forked` forks the caller itself: no server and no pickling of the call
(the child inherits it), only of the result.  It is for a caller that has
started no library thread pool -- has computed no geometry in-process and
built no PreparedSide, e.g. a worker that only dispatches calls; otherwise
the child may hang as above.  Nothing checks this: BLAS threads of numpy
are there in any process and are fork-safe, so a thread count tells nothing.
"""

from __future__ import annotations

import atexit
import ctypes
import importlib
import json
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Connection

_PR_SET_PDEATHSIG = 1
# imported by the server, so a child does not pay for them on every call;
# all but the first may be missing
_PRELOAD = ("volumetric_iou", "volumetric_iou.body", "volumetric_iou._cgal",
            "volumetric_iou.cells", "volumetric_iou.seams", "volumetric_iou.tjunction",
            "scipy.sparse", "scipy.sparse.csgraph", "scipy.spatial", "networkx")
START_TIMEOUT = 300.0            # seconds for the server to start (imports on NFS)
_PACKAGE = __name__.rpartition(".")[0]
_SERVER_CODE = ("import sys, json; sys.path[:] = json.loads(sys.argv[1]); "
                "from volumetric_iou.isolate import _serve; _serve(int(sys.argv[2]))")


# --------------------------------------------------------------------------- #
# caller side
# --------------------------------------------------------------------------- #

class _Server:
    """The fork server of this process: its pid, socket and package file."""

    def __init__(self):
        if not sys.executable:
            raise RuntimeError("timeout: cannot start the isolation server "
                               "(sys.executable is empty)")
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", _SERVER_CODE,
                 json.dumps([os.fspath(p) for p in sys.path if isinstance(p, (str, os.PathLike))]),
                 str(theirs.fileno())],
                pass_fds=(theirs.fileno(),), stdin=subprocess.DEVNULL)
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
        self.sock = ours
        self.lock = threading.Lock()
        try:
            ours.settimeout(START_TIMEOUT)
            line = b""
            while not line.endswith(b"\n"):
                chunk = ours.recv(4096)
                if not chunk:
                    raise RuntimeError(
                        f"timeout: the isolation server ({sys.executable}) exited with "
                        f"code {self.proc.wait()} before it was ready; its error is on stderr")
                line += chunk
            ours.settimeout(None)
        except socket.timeout:
            self.close()
            raise RuntimeError(f"timeout: the isolation server did not start in "
                               f"{START_TIMEOUT:.0f} s") from None
        except BaseException:
            self.close()
            raise
        hello = json.loads(line)
        self.pid, self.file, self.threads = hello["pid"], hello["file"], hello["threads"]

    def child(self):
        """(connection to a new child, its pid)."""
        mine, its = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with self.lock:
                socket.send_fds(self.sock, [b"c"], [its.fileno()])
        except BaseException:
            mine.close()
            raise
        finally:
            its.close()
        conn = Connection(mine.detach())
        try:
            pid, = struct.unpack("i", conn.recv_bytes())
        except BaseException:
            conn.close()
            raise
        return conn, pid

    def close(self):
        self.sock.close()          # the server exits on EOF
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


_server_lock = threading.Lock()
_server: _Server | None = None


def _get_server() -> _Server:
    global _server
    want = sys.modules[_PACKAGE].__file__
    with _server_lock:
        s = _server
        if s is not None and (s.proc.poll() is not None or s.file != want):
            _server = None
            s.close()
            s = None
        if s is None:
            s = _Server()
            if s.file != want:
                s.close()
                raise RuntimeError(
                    f"timeout: the isolation server imported {_PACKAGE} from {s.file}, "
                    f"this process uses {want} (not on sys.path?)")
            _server = s
        return s


def _drop_server_after_fork():
    # a forked caller gets its own server; the parent's socket is not its own
    global _server, _server_lock
    _server_lock = threading.Lock()
    s, _server = _server, None
    if s is not None:
        s.sock.close()


def _close_server():
    s = _server
    if s is not None:
        s.close()


os.register_at_fork(after_in_child=_drop_server_after_fork)
atexit.register(_close_server)


def _start_child():
    """Start a child; a server that has died since the last call is restarted once."""
    for attempt in (0, 1):
        s = _get_server()
        try:
            return s.child()
        except (OSError, EOFError):
            if attempt or s.proc.poll() is None:
                raise
    raise AssertionError("unreachable")


def _exit_code(conn, pid):
    """The child's exit code from its server: wait 1 s, SIGTERM, 1 s, SIGKILL."""
    for sig, wait in ((None, 1.0), (signal.SIGTERM, 1.0), (signal.SIGKILL, None)):
        if sig is not None:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if conn.poll(wait):
            msg = pickle.loads(conn.recv_bytes())
            if msg[0] == "exit":
                return msg[1]
    return None


def run_isolated(fn, args, kwargs, timeout: float) -> dict:
    """fn(*args, **kwargs) in a child -> {"status": ok|error|timeout|died, ...}.

    Raises if the call does not pickle or the server cannot start."""
    call = pickle.dumps((fn, args, kwargs), pickle.HIGHEST_PROTOCOL)
    main = None
    if getattr(fn, "__module__", None) in ("__main__", "__mp_main__"):
        main = getattr(sys.modules["__main__"], "__file__", None)
    head = pickle.dumps((list(sys.path), main))
    conn, pid = _start_child()
    out: dict = {"status": "died"}
    code = None
    try:
        try:
            conn.send_bytes(head)
            conn.send_bytes(call)
        except OSError:
            pass                        # the child died before it read the call
        if conn.poll(timeout):
            msg = pickle.loads(conn.recv_bytes())
            if msg[0] == "exit":
                code = msg[1]
            else:
                out = {"status": msg[0], "value": msg[1]}
        else:
            out = {"status": "timeout", "timeout": timeout}
        if out["status"] in ("died", "timeout") and code is None:
            code = _exit_code(conn, pid)
    except EOFError:
        pass                            # the server died
    finally:
        conn.close()
    if code is not None and code < 0 and out["status"] in ("died", "timeout"):
        out["signal"] = signal.Signals(-code).name
    return out


def run_forked(fn, args, kwargs, timeout: float) -> dict:
    """fn(*args, **kwargs) in a child forked from this process -> as run_isolated.

    Only for a caller without library thread pools (module docstring).  The
    first call imports what the server would (imports start no pool), so a
    child does not import it again on every call."""
    global _preloaded
    if not _preloaded:
        _preload()
        _preloaded = True
    _reap_done()
    from multiprocessing import Pipe
    r, w = Pipe(duplex=False)
    parent = os.getpid()
    _flush()                            # else the child writes the caller's buffered output again
    pid = os.fork()
    if pid == 0:
        r.close()
        _forked_child(w, parent, lambda: fn(*args, **kwargs))
    w.close()                           # otherwise EOF from a dead child never arrives
    out: dict = {"status": "died"}
    try:
        if r.poll(timeout):
            status, value = pickle.loads(r.recv_bytes())
            out = {"status": status, "value": value}
        else:
            out = {"status": "timeout", "timeout": timeout}
    except EOFError:
        pass                            # the child died
    finally:
        r.close()
    if out["status"] in ("ok", "error"):
        _exited.append(pid)             # it exits on its own; reaped at a later call
        return out
    code = _reap_forked(pid)
    if code is not None and code < 0 and out["status"] in ("died", "timeout"):
        out["signal"] = signal.Signals(-code).name
    return out


_preloaded = False
_exited: list[int] = []                 # forked children that have replied, not reaped yet


def _reap_done() -> None:
    for pid in list(_exited):
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            done = pid
        if done:
            _exited.remove(pid)


def _preload() -> None:
    for name in _PRELOAD:
        try:
            importlib.import_module(name)
        except ImportError:
            if name == _PRELOAD[0]:
                raise


def _reap_forked(pid):
    """The exit code of a forked child: wait 1 s, SIGTERM, 1 s, SIGKILL."""
    # polled: os.pidfd_open is missing in some builds (conda's python 3.12)
    try:
        for sig, wait in ((None, 1.0), (signal.SIGTERM, 1.0), (signal.SIGKILL, None)):
            if sig is not None:
                os.kill(pid, sig)       # not reaped yet: the pid is still the child's
            if wait is None:
                return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
            end, pause = time.monotonic() + wait, 1e-4     # a child that has replied exits at once
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    return os.waitstatus_to_exitcode(status)
                if time.monotonic() > end:
                    break
                time.sleep(pause)
                pause = min(2 * pause, 0.01)
    except ChildProcessError:
        pass                            # reaped elsewhere (the caller's SIGCHLD handler)
    return None


def _forked_child(conn, parent_pid, call) -> None:
    code = 1
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() != parent_pid:
            os._exit(1)
        conn.send_bytes(_result(call))
        code = 0
    finally:
        _flush()
        os._exit(code)


def _result(call) -> bytes:
    """The pickled ("ok", value) of call(), or ("error", message)."""
    try:
        out = ("ok", call())
    except BaseException as e:          # noqa: BLE001 -- report any failure upward
        out = ("error", f"{type(e).__name__}: {e}")
    try:
        return pickle.dumps(out, pickle.HIGHEST_PROTOCOL)
    except BaseException as e:          # noqa: BLE001
        return pickle.dumps(("error", f"{type(e).__name__}: {e}"))


def _flush() -> None:
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except BaseException:           # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# server side
# --------------------------------------------------------------------------- #

def _serve(fd: int) -> None:
    """The server loop: a child per received socket, its exit code sent back on it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _preload()
    ctrl = socket.socket(fileno=fd)
    wake_r, wake_w = os.pipe()
    os.set_blocking(wake_r, False)
    os.set_blocking(wake_w, False)
    signal.signal(signal.SIGCHLD, lambda *_: None)
    signal.set_wakeup_fd(wake_w)
    me = os.getpid()
    ctrl.sendall(json.dumps({"pid": me, "file": sys.modules[_PACKAGE].__file__,
                             "threads": len(os.listdir("/proc/self/task"))}).encode() + b"\n")
    children: dict[int, Connection] = {}
    while True:
        ready, _, _ = select.select([ctrl, wake_r], [], [])
        if wake_r in ready:
            try:
                while os.read(wake_r, 4096):
                    pass
            except BlockingIOError:
                pass
            _reap(children)
        if ctrl in ready:
            msg, fds, _, _ = socket.recv_fds(ctrl, 16, 4)
            if not msg:
                break                   # the caller has closed its end or died
            for extra in fds[1:]:
                os.close(extra)
            if not fds:
                continue
            conn = Connection(fds[0])
            pid = os.fork()
            if pid == 0:
                _child(conn, me, [ctrl.detach(), wake_r, wake_w],
                       [c for c in children.values()])
            children[pid] = conn
            try:
                conn.send_bytes(struct.pack("i", pid))
            except OSError:
                pass
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _reap(children: dict) -> None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        conn = children.pop(pid, None)
        if conn is not None:
            try:
                conn.send_bytes(pickle.dumps(("exit", os.waitstatus_to_exitcode(status))))
            except OSError:
                pass                    # the caller has stopped listening
            conn.close()


def _child(conn, server_pid, fds, others) -> None:
    code = 1
    try:
        signal.set_wakeup_fd(-1)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        for f in fds:
            os.close(f)
        for c in others:
            c.close()
        # dies with the server (the server is single-threaded, so its thread
        # that forked lives as long as it does); the check covers a server
        # that died before prctl
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() != server_pid:
            os._exit(1)
        path, main = pickle.loads(conn.recv_bytes())
        sys.path[:] = path

        def call():
            if main is not None:
                from multiprocessing import spawn
                spawn.import_main_path(main)
            fn, args, kwargs = pickle.loads(conn.recv_bytes())
            return fn(*args, **kwargs)

        conn.send_bytes(_result(call))
        code = 0
    finally:
        _flush()
        os._exit(code)
