import torch
from scipy.spatial import distance_matrix

def sim_matrix(a, b, S_lambda, inv_temp, eps=1e-8):  # matrix cosines
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    # torch.save(sim_mt, 'cosine_similarity _matrix.pt')
    spiky_mt = sim_mt**inv_temp
    spiky_mt = spiky_mt/(spiky_mt.sum(1)[:,None])
    sim_mt = torch.eye(sim_mt.shape[0]).to(sim_mt.device) + S_lambda*(spiky_mt) # spiky distribution

    return sim_mt

def gaussian_distance_matrix(a, b, S_lambda, inv_temp, eps=1e-8, read_from_file = True):
    """
    added eps for numerical stability
    """
    if read_from_file:
        
        gaussian_dist = torch.load('prob_matrix.pt').to('cpu')
    
    else:

        gaussian_dist = distance_matrix(a.detach().cpu(),b.detach().cpu())

        torch.save(gaussian_dist, 'gaussian_distance_matrix.pt')
    
        gaussian_dist = torch.exp(torch.tensor(-1.0*(gaussian_dist**2)))

        norm_gaussian_dist = gaussian_dist.sum(dim=1)[:,None]

        gaussian_dist = gaussian_dist/norm_gaussian_dist

    gaussian_dist = S_lambda*gaussian_dist

    spiky_mt = gaussian_dist**inv_temp

    spiky_mt = spiky_mt/(spiky_mt.sum(1)[:,None])
    
    gaussian_dist = torch.eye(gaussian_dist.shape[0]).to(gaussian_dist.device) + spiky_mt # spiky distribution

    return gaussian_dist