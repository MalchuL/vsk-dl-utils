import os
from pathlib import Path
from typing import List

from natsort import natsorted
from tqdm import tqdm

IMG_EXTENSIONS = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# TODO cover by tests
def iterate_with_creating_structure(
    in_folder: str, out_folder: str, supported_extensions: List[str] = IMG_EXTENSIONS
):
    """Iterates over files and returns files with same folder structure.

    Output folders creates automatically.
    :param in_folder: Folder to iterate.
    :param out_folder: Folder to save images.
    :param supported_extensions: Files extensions to iterate.
    :return: iterator with filepath and output path. If output path doesn't exist it will be
        created automatically when meet file.
    """
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    files = []
    for pattern in supported_extensions:
        files.extend(in_folder.rglob(pattern="*" + pattern))

    for file_path in tqdm(natsorted(files)):
        sub_path = os.path.relpath(file_path, in_folder)
        new_path = out_folder / sub_path
        if not os.path.exists(new_path.parent):
            new_path.parent.mkdir(exist_ok=True, parents=True)  # Create parent folder

        yield (file_path, new_path)


def iterate_recursively(in_folder, supported_extensions=IMG_EXTENSIONS):
    in_folder = Path(in_folder)
    files = []
    for pattern in supported_extensions:
        files.extend(in_folder.rglob(pattern="*" + pattern))

    yield from files
