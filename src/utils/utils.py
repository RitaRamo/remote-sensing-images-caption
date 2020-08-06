from torch.nn.utils.rnn import pack_padded_sequence
import torch
from torch.autograd import Variable
from sklearn.model_selection import ParameterSampler


def cdist(x, y):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    return sim_mt


def get_pack_padded_sequences(predictions, targets, caption_lengths):
    predictions = pack_padded_sequence(
        predictions, caption_lengths, batch_first=True)
    targets = pack_padded_sequence(
        targets, caption_lengths, batch_first=True)

    return predictions.data, targets.data

# not my code -> it was taken from: PyTorchOT (https://github.com/rythei/PyTorchOT)


def sink(M, reg, numItermax=1000, stopThr=1e-9, cuda=True):

    # we assume that no distances are null except those of the diagonal of
    # distances

    a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
    b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()

    # init data
    Nini = len(a)
    Nfin = len(b)

    u = Variable(torch.ones(Nini) / Nini).cuda()
    v = Variable(torch.ones(Nfin) / Nfin).cuda()

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)

            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)


def get_weights_with_parameter_sampler():
    param_grid = dict(c2=[round(x * 0.1, 1) for x in range(0, 11)],
                      c3=[round(x * 0.1, 1) for x in range(0, 11)],
                      c4=[round(x * 0.1, 1) for x in range(0, 11)])
    param_list = list(ParameterSampler(param_grid, n_iter=15, random_state=42))
    return param_list
