import os


class Directory:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    DOWNLODAD_DIR = os.path.join(ROOT_DIR, 'download')
    TRANSFORM_DIR = os.path.join(ROOT_DIR, 'transform')