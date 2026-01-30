import tkinter as tk
from tkinter import filedialog
import shutil
from pathlib import Path
from loguru import logger


def fetch_user_dir(title:str, base_directory:Path=Path.home()):
    '''
    GUI in which user select phone's data
    '''
    root = tk.Tk()
    root.withdraw()

    selected_dir = filedialog.askdirectory(
        parent=root,
        initialdir=str(base_directory),
        mustexist=True,
        title=title
    )
    root.destroy()

    if selected_dir:
        return Path(selected_dir)
    else:
        logger.warning("Directory not selected")
        return None
    

def found_images_list(root:Path) -> list:
    '''
    From Path found all images in this path and its children
    Returns: list of path of imgs
    '''
    image_exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    found_images = []
    for root, _, files in root.walk():
        for file in files:
            if file.lower().endswith(image_exts):
                path = root / file
                if path.is_file():
                    found_images.append(path)

    return found_images


def move_to_folder(
        input:Path, dest:Path, img_list:list, overwrite:bool=False
    ):
    '''
    Create a 'bin' folder where all images from phone will be placed
    (The user will delete manually them after all)
    Then copy preserving metadata to destination folder and move
    the photo in bin folder
    '''
    logger.info('Create folder where images will be stored')
    bin = input / 'To be eliminated'
    bin.mkdir(parents=True, exist_ok=True)

    for path in img_list:
        target = dest / path.name
        img_to_bin = bin / path.name
        # Counter if there's already the file
        i = 1
        while target.exists() and not overwrite:
            name, suffix = path.stem, path.suffix
            target = dest / f"{name}_{i}{suffix}"
            i += 1
        # Create a copy to destination, move to 'bin' the original
        shutil.copy2(path, target)
        path.replace(img_to_bin)
    logger.success('All images moved')
