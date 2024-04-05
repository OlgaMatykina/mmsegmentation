# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NkbRobosegmentSmallDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('unlabelled',
                'firehose',
                'hose',
                'waste',
                'puddle',
                'breakroad',
                'sidewalk',
                'terrain',
                'vegetation',
                'road'),
        palette=[[255,255,255], #'unlabelled' : none
                [255,0,0], #'firehose' : red
                [255,165,0], #'hose' : orange
                [0,0,255], #'waste' : blue
                [255,255,0], #'puddle' : yellow
                [0,255,255], #'breakroad' : aqua
                [255,0,255], #'sidewalk' : magenta
                [0,128,0], #'terrain': green
                [127,72,41], #'vegetation': brown
                [250,128,114]
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
