"""Access to code from the ReMake repository (MIT), which also ships the TDCNet model code.

We do not vendor that code. `scripts/setup_third_party.sh` clones it into ``third_party/ReMake``
(or set the ``REMAKE_REPO`` environment variable). Its top-level folders are named ``models``,
``utils`` and ``datasets``, which clash with installed packages (e.g. Hugging Face ``datasets``),
so we import them in an isolated way.
"""
import os
import re
import sys
import importlib

_CLASH = ("models", "utils", "datasets", "run_utils", "run_tools", "relat_depth_models")
_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "ReMake")


def remake_dir():
    d = os.environ.get("REMAKE_REPO", _DEFAULT)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"ReMake repository not found at {d}. Run scripts/setup_third_party.sh "
                                "or set REMAKE_REPO=/path/to/ReMake")
    return d


def _prepare(repo):
    # make the repository folders real packages so they win over same-named installed packages
    for name in _CLASH:
        d = os.path.join(repo, name)
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, "__init__.py")):
            open(os.path.join(d, "__init__.py"), "w").close()
    for k in [k for k in list(sys.modules) if k.split(".")[0] in _CLASH]:
        sys.modules.pop(k)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def import_from_remake(module, attr):
    """e.g. import_from_remake('models.tdcnet', 'TDCNet')"""
    repo = remake_dir()
    _prepare(repo)
    cwd = os.getcwd()
    os.chdir(repo)              # some modules read relative paths at import time
    try:
        return getattr(importlib.import_module(module), attr)
    finally:
        os.chdir(cwd)


class in_remake_dir:
    """Run code with the ReMake repository as working directory (its modules use relative paths)."""

    def __enter__(self):
        self.cwd = os.getcwd()
        os.chdir(remake_dir())

    def __exit__(self, *exc):
        os.chdir(self.cwd)


def TDCNet():
    return import_from_remake("models.tdcnet", "TDCNet")


def surface_normal_fn():
    return import_from_remake("utils.tools", "get_surface_normal_from_xyz")


def patch_numpy2():
    """The official TransCG augmentation code calls int(np.random.randint(n, size=1)), which fails on NumPy 2.
    Drop `size=1` so the call returns a scalar. Idempotent; only touches the cloned third-party file."""
    path = os.path.join(remake_dir(), "utils", "data_preparation.py")
    src = open(path).read()
    new, k = re.subn(r"\b(int|float)\((np\.random\.\w+\([^()]*?(?:\([^()]*\)[^()]*?)*?),\s*size\s*=\s*1\)\)", r"\1(\2))", src)
    if k:
        open(path, "w").write(new)
    return k


def official_transcg(data_dir, split="train", **kw):
    """The official TransCG loader (as shipped in the ReMake repo), used for depth-model training."""
    cls = import_from_remake("datasets.transcg", "TransCG")
    return cls(data_dir, split=split, **kw)
