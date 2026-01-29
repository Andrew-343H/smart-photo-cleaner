import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from loguru import logger


def fetch_input_dir(base_directory=Path.home()):
    '''
    GUI in which user select phone's data
    '''
    root = tk.Tk()
    root.withdraw()

    selected_dir = filedialog.askdirectory(
        parent=root,
        initialdir=str(base_directory),
        mustexist=True,
        title="Select main folder"
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
