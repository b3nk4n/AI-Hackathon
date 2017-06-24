"""Utility module for directory of file path functions."""
import os
import fnmatch


def get_filenames(root_dir, pattern, include_root=True):
    """Gets a list of files of a given directory matching a
       specified pattern, by using a resursive search.
    Parameters
    ----------
    root_dir: str
        The directory to recursively look into.
    pattern: str
        The file pattern search string, such as '*.jpg'.
    include_root: Boolean, optional
        Whether to include the root path or just return the
        filenames.
    Returns
    ----------
    matches: list(string)
        Returns a list of filenames that match this pattern.
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            if include_root:
                filename = os.path.join(root, filename)
            matches.append(filename)
    return matches
