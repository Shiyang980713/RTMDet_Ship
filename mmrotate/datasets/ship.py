# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class ShipDataset(BaseDataset):
    """ShipDataset dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
        ('Nimitz Aircraft Carrier', 'Barracks Ship', 'Container Ship', 'Fishing Vessel', 'Henry J. Kaiser-class replenishment oiler', 'Other Warship', 'Yacht', 'Freedom-class littoral combat ship', 'Arleigh Burke-class Destroyer', 'Lewis and Clark-class dry cargo ship', 'Towing vessel', 'unknown', 'Powhatan-class tugboat', 'Barge', '055-destroyer', '052D-destroyer', 'USNS Bob Hope', 'USNS Montford Point', 'Bunker', 'Ticonderoga-class cruiser', 'Oliver Hazard Perry-class frigate', 'Sacramento-class fast combat support ship', 'Submarine', 'Emory S. Land-class submarine tender', 'Hatakaze-class destroyer', 'Murasame-class destroyer', 'Whidbey Island-class dock landing ship', 'Hiuchi-class auxiliary multi-purpose support ship', 'USNS Spearhead', 'Hyuga-class helicopter destroyer', 'Akizuki-class destroyer', 'Bulk carrier', 'Kongo-class destroyer', 'Northampton-class tug', 'Sand Carrier', 'Iowa-class battle ship', 'Independence-class littoral combat ship', 'Tarawa-class amphibious assault ship', 'Cyclone-class patrol ship', 'Wasp-class amphibious assault ship', '074-landing ship', '056-corvette', '721-transport boat', '037II-missile boat', 'Traffic boat', '037-submarine chaser', 'unknown auxiliary ship', '072III-landing ship', '636-hydrographic survey ship', '272-icebreaker', '529-Minesweeper', '053H2G-frigate', '909A-experimental ship','909-experimental ship', '037-hospital ship', 'Tuzhong Class Salvage Tug', '022-missile boat','051-destroyer', '054A-frigate','082II-Minesweeper', '053H1G-frigate', 'Tank ship', 'Hatsuyuki-class destroyer', 'Sugashima-class minesweepers', 'YG-203 class yard gasoline oiler', 'Hayabusa-class guided-missile patrol boats', 'JS Chihaya', 'Kurobe-class training support ship', 'Abukuma-class destroyer escort', 'Uwajima-class minesweepers', 'Osumi-class landing ship', 'Hibiki-class ocean surveillance ships', 'JMSDF LCU-2001 class utility landing crafts', 'Asagiri-class Destroyer', 'Uraga-class Minesweeper Tender', 'Tenryu-class training support ship', 'YW-17 Class Yard Water', 'Izumo-class helicopter destroyer', 'Towada-class replenishment oilers', 'Takanami-class destroyer', 'YO-25 class yard oiler','891A-training ship', '053H3-frigate', '922A-Salvage lifeboat', '680-training ship', '679-training ship','072A-landing ship', '072II-landing ship', 'Mashu-class replenishment oilers', '903A-replenishment ship','815A-spy ship', '901-fast combat support ship', 'Xu Xiake barracks ship', 'San Antonio-class amphibious transport dock', '908-replenishment ship', '052B-destroyer', '904-general stores issue ship', '051B-destroyer', '925-Ocean salvage lifeboat','904B-general stores issue ship', '625C-Oceanographic Survey Ship', '071-amphibious transport dock', '052C-destroyer', '635-hydrographic Survey Ship', '926-submarine support ship', '917-lifeboat', 'Mercy-class hospital ship', 'Lewis B. Puller-class expeditionary mobile base ship', 'Avenger-class mine countermeasures ship', 'Zumwalt-class destroyer', '920-hospital ship','052-destroyer', '054-frigate', '051C-destroyer', '903-replenishment ship', '073-landing ship', '074A-landing ship', 'North Transfer 990', '001-aircraft carrier', '905-replenishment ship', 'Hatsushima-class minesweeper', 'Forrestal-class Aircraft Carrier', 'Kitty Hawk class aircraft carrier', 'Blue Ridge class command ship','081-Minesweeper','648-submarine repair ship', '639A-Hydroacoustic measuring ship', 'JS Kurihama', 'JS Suma', 'Futami-class hydro-graphic survey ships', 'Yaeyama-class minesweeper', '815-spy ship', 'Sovremenny-class destroyer'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
           (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
           (0, 128, 128), (0, 0, 128), (0, 0, 70), (70, 70, 70), (70, 130, 180), (100, 149, 237),
           (240, 230, 140), (255, 215, 0), (218, 165, 32), (184, 134, 11), (188, 143, 143), (205, 92, 92),
           (139, 69, 19), (160, 82, 45), (255, 127, 80), (255, 99, 71), (255, 0, 0), (165, 42, 42),
           (178, 34, 34), (240, 128, 128), (233, 150, 122), (250, 128, 114), (255, 160, 122), (255, 165, 0),
           (255, 140, 0), (255, 97, 3), (255, 69, 0), (255, 127, 36), (224, 255, 255), (240, 255, 255),
           (245, 255, 250), (240, 248, 255), (248, 248, 255), (240, 255, 240), (255, 240, 245), (255, 248, 220),
           (253, 245, 230), (245, 245, 220), (255, 250, 205), (255, 245, 238), (245, 255, 250), (112, 128, 144),
           (220, 220, 220), (211, 211, 211), (131, 111, 255), (72, 61, 139), (123, 104, 238), (106, 90, 205),
           (238, 232, 170), (255, 250, 240), (255, 228, 196), (255, 228, 181), (255, 222, 173), (245, 222, 179),
           (222, 184, 135), (210, 180, 140), (188, 143, 143), (244, 164, 96), (199, 21, 133), (218, 112, 214),
           (255, 105, 180), (255, 20, 147), (255, 182, 193), (255, 160, 122), (32, 178, 170), (135, 206, 235),
           (255, 228, 196), (255, 217, 179), (255, 218, 185), (255, 222, 173), (255, 228, 181),
           (255, 248, 220), (220, 220, 220), (211, 211, 211), (188, 188, 188), (169, 169, 169),
           (128, 128, 128), (105, 105, 105), (119, 136, 153), (112, 128, 144), (255, 250, 250),
           (255, 245, 245), (220, 220, 220), (255, 250, 240), (240, 255, 240), (255, 240, 240),
           (255, 255, 240), (240, 248, 255), (248, 248, 255), (245, 245, 245), (255, 255, 255),
           (245, 255, 250), (240, 255, 255), (240, 248, 255), (230, 230, 250), (255, 240, 245),
           (255, 228, 225), (255, 222, 173), (255, 228, 181), (255, 235, 205), (245, 245, 220),
           (245, 222, 179), (220, 220, 220), (188, 188, 188), (128, 128, 128), (105, 105, 105),
           (132, 112, 255), (0, 0, 139), (0, 0, 128), (25, 25, 112), (255, 255, 0), (154, 205, 50),
           (34, 139, 34), (50, 205, 50), (144, 238, 144), (152, 251, 152), (143, 188, 143),
           (240, 230, 140), (238, 232, 170), (189, 183, 107), (202, 255, 112)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'bmp',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split(',')
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
