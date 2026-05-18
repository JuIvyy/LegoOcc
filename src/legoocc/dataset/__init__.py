import torch
import numpy as np
from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from .dataset_wrapper_scannet_occ import Scannet_Scene_Occ_DatasetWrapper
from .dataset_scannet_occ_openocc import Scannet_Scene_OpenOccupancy_Dataset
# from .dataset_scannet_online_occ import Scannet_Online_SceneOcc_Dataset
# from .dataset_wrapper_scannet_online import Scannet_Online_SceneOcc_DatasetWrapper


def custom_collate_fn(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, np.ndarray):
            data_tuple.append(torch.from_numpy(np.stack([d[i] for d in data])))
        elif isinstance(item, str):
            data_tuple.append([d[i] for d in data])
        elif isinstance(item, dict):
            # collated_dict = {}
            # for key in item.keys():
            #     key_values = [d[i][key] for d in data]
            #     if isinstance(item[key], np.ndarray):
            #         # print(key, [v.shape for v in key_values])
            #         try:
            #             collated_dict[key] = torch.from_numpy(np.stack(key_values))
            #         except ValueError as e:
            #             collated_dict[key] = key_values

            #     elif isinstance(item[key], torch.Tensor):
            #         collated_dict[key] = torch.stack(key_values)
            #     elif isinstance(item[key], (str, dict)):
            #         collated_dict[key] = key_values
            #     elif item[key] is None:
            #         collated_dict[key] = [None for _ in data]
            #     else:
            #         collated_dict[key] = key_values
            # data_tuple.append(collated_dict)
            data_tuple.append([d[i] for d in data])

        elif item is None:
            data_tuple.append([None for _ in data])
        elif isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        else:
            raise TypeError(f"{type(item)} is not supported")
    return data_tuple


# def custom_collate_fn(data):
#     num_elements = len(data[0]) if isinstance(data[0], (list, tuple)) else 1
    
#     collated_data = []
    
#     for element_idx in range(num_elements):
#         elements = [sample[element_idx] if isinstance(sample, (list, tuple)) else sample for sample in data]

#         first_element = elements[0]
        
#         if isinstance(first_element, np.ndarray):
#             collated_data.append(torch.from_numpy(np.stack(elements)))
#         elif isinstance(first_element, torch.Tensor):
#             collated_data.append(torch.stack(elements))
#         elif isinstance(first_element, (dict, str)):
#             collated_data.append(elements)
#         elif first_element is None:
#             collated_data.append(None)
#         else:
#             raise TypeError(f"{type(first_element)} is not supported")
    
#     return collated_data


def build_dataloader(
            train_dataset_config,
            val_dataset_config,
            train_wrapper_config,
            val_wrapper_config,
            train_loader_config,
            val_loader_config,
            dist=False,
    ):
    train_dataset = OPENOCC_DATASET.build(train_dataset_config)
    val_dataset = OPENOCC_DATASET.build(val_dataset_config)

    train_wrapper = OPENOCC_DATAWRAPPER.build(train_wrapper_config, default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(val_wrapper_config, default_args={'in_dataset': val_dataset})

    train_sampler = val_sampler = None
    if dist:
        train_sampler = DistributedSampler(train_wrapper, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_wrapper, shuffle=False, drop_last=False)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader_config["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False if dist else train_loader_config["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader_config["num_workers"],
        persistent_workers=train_loader_config["num_workers"] > 0,
        pin_memory=True)
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader_config["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False if dist else val_loader_config["shuffle"],
        sampler=val_sampler,
        persistent_workers=val_loader_config["num_workers"] > 0,
        num_workers=val_loader_config["num_workers"],
        pin_memory=True)

    return train_dataset_loader, val_dataset_loader