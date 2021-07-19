from .blender import BlenderDataset
from .llff import LLFFDataset
from .nusc import NusDataset
from .llff2d import LLFF2DDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'nusc': NusDataset,
                'llff2d': LLFF2DDataset}