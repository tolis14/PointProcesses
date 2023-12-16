import torch

def get_folders(N: int, k: int, r:int):
    """
    :param N: number of training examples
    :param k: number of folders
    :param r: number of replications
    :return: k-folders with r-replications
    The output of this function looks like
    [
    [[f_1], [f_2],...,[f_k]] <-- replication 1
     [f_1], [f_2],...,[f_k]] <-- replication 2
                  ...
                  ...
     [f_1], [f_2],...,[f_k]] <-- replication r
    ]
    """
    replications = []
    for _ in range(r):
        random_indices = torch.randperm(N) # shuffle the indices
        folders = [[] for _ in range(k)] # make k-folders for this replication
        # assign indices into folders
        # in round robin fashion
        for i in range(len(random_indices)):
            folders[i % k] += [random_indices[i].item()]
        replications.append(folders)
    return replications