import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
from pathlib import Path
from loguru import logger
import yaml
import sys


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
    for dirpath, _, files in root.walk():
        for file in files:
            if file.lower().endswith(image_exts):
                path = dirpath / file
                if path.is_file():
                    found_images.append(path)

    return found_images

def add_more():
    message="Do you want to select other folder?"
    result = messagebox.askyesno("Add more", message)
    return result


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


def extract_config(config_name: str, same_lv: bool = False) -> dict:
    """
    Load configuration data from a configuration file.
    Params:
    - config_name: the name of yaml file
    - samelv: identify if it belongs to folder name Risorse or to main folder
    """
    # Support functions
    def get_application_root() -> Path:
        # Executed as .exe
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        # Executed as script
        return Path(__file__).resolve().parent.parent

    def build_config_path(
        config_name: str, same_level: bool, app_root: Path
    ) -> Path:
        # In case .exe is in the folder of config
        if same_level:
            return app_root / config_name
        
        return app_root.parent / "config" / config_name

    def load_yaml_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        if not isinstance(config, dict):
            raise ValueError("Configuration file is not a valid YAML mapping")

        return config

    app_root = get_application_root()
    config_path = build_config_path(config_name, same_lv, app_root)
    config = load_yaml_config(config_path)
    
    return config


