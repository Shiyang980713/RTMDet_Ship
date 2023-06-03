# Copyright (c) OpenMMLab. All rights reserved.
from .dota_metric import DOTAMetric
from .ship_metric import ShipMetric
from .rotated_coco_metric import RotatedCocoMetric

__all__ = ['DOTAMetric', 'RotatedCocoMetric', 'ShipMetric']
