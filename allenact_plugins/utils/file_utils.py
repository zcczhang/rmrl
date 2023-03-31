import errno
import glob
import json
import os
import shutil


__all__ = [
    "f_expand",
    "f_join",
    "f_listdir",
    "f_mkdir",
    "f_remove",
    "load_json",
    "dump_json",
]


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))


def f_exists(*fpaths):
    return os.path.exists(f_join(*fpaths))


def f_join(*fpaths):
    """join file paths and expand special symbols like `~` for home dir."""
    return f_expand(os.path.join(*fpaths))


def f_listdir(*fpaths, filter=None, sort=False, full_path=False, nonexist_ok=True):
    """
    Args:
        full_path: True to return full paths to the dir contents
        filter: function that takes in file name and returns True to include
        nonexist_ok: True to return [] if the dir is non-existent, False to raise
        sort: sort the file names by alphabetical
    """
    dir_path = f_join(*fpaths)
    if not os.path.exists(dir_path) and nonexist_ok:
        return []
    files = os.listdir(dir_path)
    if filter is not None:
        files = [f for f in files if filter(f)]
    if sort:
        files.sort()
    if full_path:
        return [os.path.join(dir_path, f) for f in files]
    else:
        return files


def f_mkdir(*fpaths):
    """Recursively creates all the subdirs If exist, do nothing."""
    fpath = f_join(*fpaths)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def f_remove(fpath, verbose=False, dry_run=False):
    """If exist, remove.

    Supports both dir and file. Supports glob wildcard.
    """
    assert isinstance(verbose, bool)
    fpath = f_expand(fpath)
    if dry_run:
        print("Dry run, delete:", fpath)
        return
    for f in glob.glob(fpath):
        try:
            shutil.rmtree(f)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                try:
                    os.remove(f)
                except:  # final resort safeguard
                    pass
    if verbose:
        print(f'Deleted "{fpath}"')


def load_json(*file_path, **kwargs):
    file_path = f_join(*file_path)

    with open(file_path, "r") as fp:
        return json.load(fp, **kwargs)


def dump_json(data, *file_path, convert_to_primitive=False, **kwargs):
    if convert_to_primitive:
        from .array_tensor_utils import any_to_primitive

        data = any_to_primitive(data)
    file_path = f_join(*file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)
