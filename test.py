from torch.distributions import Normal, Independent
import torch

q = Independet(Normal(
    loc=torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    scale=torch.tensor([[1, 1, 1, 1, 1], [2, 1, 2, 1, 2]])
), 1)

p = Independet(Normal(
    loc=torch.tensor([[0, 1.5, 2, 3, 3.5], [6, 6, 6, 7, 9]]),
    scale=torch.tensor([[1, 2, 1, 1, 2], [2, 2, 2, 1, 0.5]])
), 1)

x = q.rsample()
print(q.log_prob(x) - p.log_prob(x))

kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
kl = kl - 1 + p_logsigma - q_logsigma
kl = 0.5 * torch.sum(kl, dim=-1)

print(kl)