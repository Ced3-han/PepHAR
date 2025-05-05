import pickle
from ._base import register_dataset
from torch.utils.data import Dataset

class ProteinPeptideDataset(Dataset):
    memorizing_flag = True
    memorizing_dict = {}

    def __init__(self, root, train=True, split_ratio=1.0, transform=None, return_name=False):
        # if self.memorizing_flag and root in self.memorizing_dict:
        #     dict_data = self.memorizing_dict[root]
        # else:
        #     with open(f'{root}/data.pkl', "rb") as file:
        #         dict_data = pickle.load(file)
        #     self.memorizing_dict[root] = dict_data
        if train:
            with open(root, "rb") as file:
                    data_list = pickle.load(file)
            #self.memorizing_dict[root] = dict_data
        else:
            with open(root, "rb") as file:
                    data_list = pickle.load(file)
            #self.memorizing_dict[root] = dict_data

        self.return_name = return_name
        self.data_list = data_list
        self.name_list = [data['name'] for data in data_list]
        # all_name_list = dict_data.keys()
        # sorted_name_list = sorted(all_name_list)
        # split_num = int(len(sorted_name_list) * split_ratio)
        # self.name_list = sorted_name_list[:split_num] if train else sorted_name_list[split_num:]
        # self.data_list = [dict_data[name] for name in self.name_list]
        #
        # assert len(self.name_list) == len(self.data_list)

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample if not self.return_name else (self.name_list[idx], sample)


    def get_item_by_name(self, name):
        idx = self.name_list.index(name)
        return self.__getitem__(idx)


@register_dataset('protein_peptide')
def get_protein_peptide_dataset(cfg, transform):
    return ProteinPeptideDataset(
        root=cfg.data_dir,
        train=cfg.is_train,
        split_ratio=cfg.split_ratio,
        return_name=cfg.return_name,
        transform=transform,
    )

if __name__ == '__main__':
    dataset = ProteinPeptideDataset(root='pep-design-ori/data/dataset_v3/pepdb_train_v3.pkl', train=True)
    print(len(dataset))