import torch

def save_model_realizations(realizations: dict, intensity_name: str):

    base_path = '..//..//william_results//samples2D//'

    for method, method_realizations in realizations.items():
        index = 1
        full_path = base_path + method + '_' + intensity_name + '_'
        for realization in method_realizations:
            torch.save(realization, full_path + str(index) + '.pt')
            index += 1

