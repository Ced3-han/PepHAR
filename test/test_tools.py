from utils.tools import *

def test_get_structure():
    ws_dir = '/home/chentong/ws'
    data_root = f'{ws_dir}/pep-design/data/protein_peptide_tensor'
    data = get_strucutre(data_root, ('train', 0))
    print(data)
    data = get_strucutre(data_root, '1a1o')
    print(data)

if __name__ == '__main__':
    test_get_structure()
