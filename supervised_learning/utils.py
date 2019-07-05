###
'''
June 2019
Code by: Arnaud Fickinger
'''
###

def calculate_threshold(model, rate):
    empty = torch.Tensor()
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs = expand_model(model, empty)
    weights = torch.abs(pre_abs)

    return np.percentile(weights.detach().cpu().numpy(), rate)