# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12_data_utils.ipynb (unless otherwise specified).

__all__ = ['coco_metadata', 'h36m_metadata', 'suggest_metadata']

# Cell
coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

# Cell
h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}

# Cell
def suggest_metadata(name):
    """Returns the metadata for a specific dataset."""
    names = []
    for metadata in [coco_metadata, h36m_metadata]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))