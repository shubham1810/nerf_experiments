from .blender import BlenderDataset
from .llff import LLFFDataset
from .nusc import NusDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'nusc': NusDataset}