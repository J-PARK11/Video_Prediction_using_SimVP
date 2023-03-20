from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_kth_custom import load_data as load_kth
from .dataloader_caltech import load_data as load_caltech

def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, out_frame, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kth':
        return load_kth(data_root, batch_size, val_batch_size, num_workers, out_frame)
    elif dataname == 'caltech':
        return load_caltech(data_root, batch_size, val_batch_size, num_workers, out_frame)