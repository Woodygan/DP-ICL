import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from typing import List
import torch.nn.functional as F
import numpy as np
from math import comb
import copy
from scipy.optimize import bisect, brentq
import os
import tqdm
import functools
from itertools import combinations
import multiprocessing as mp
import gc
from ngram import generate_ngrams
from scipy.special import logsumexp

class PMixED():
    def __init__(self, model, model_name, tokenizer,
                device="cpu", q_budget=30, alpha=2, delta=1e-5,
                Ex_num=3.0, eps=50, beta=0., lambd=None, threshold=None, screen_top_k=0,
                sigma=None, accounting_method=None, 
    ):
        self.device=device
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.q_budget = q_budget
        self.alpha = alpha 
        self.eps = eps
        self.priv_loss = 0
        self.Ex_num=Ex_num
        self.p = 0.0 
        self.delta = delta
        self.sigma = sigma
        self.accounting_method = accounting_method
        self.num_ensemble = 100
        self.lambd = lambd
        self.lambdas = np.array([0 for _ in range(self.num_ensemble)])
        self.lambda_history = []
        self.noisy_rd_history = []
        self.target = beta * self.alpha
        self.beta = beta
        self.threshold = threshold
        self.screen_top_k = screen_top_k
        self.num_noisy = 0 
        self.num_non_sample = 0
        self.num_noise = [] 
        self.pub_model = model.to(self.device)
        self.lora_ensemble = 0
        self.tau = 0.4 / self.alpha
        self.ls_cum = np.zeros(self.num_ensemble-2) 
        self.beta_array=[]
        self.context=[]
        self.vocab_size=self.tokenizer.vocab_size
    @staticmethod 
    def subsample_eps(alpha, p, **kwargs):
        '''
        Make priv loss a parameter 
        '''
        return 1/(alpha-1) * np.log((1-p)**(alpha-1) * (1 + (alpha-1) * p) 
                                    + np.sum([comb(alpha, k) * (1-p)**(alpha-k) * p**k *
                                              np.exp((k-1) * PMixED.data_indep_loss(alpha=alpha, **kwargs))
                                                for k in range(2, alpha+1)]
                                            )
                                        )
    @staticmethod
    def sample_privacy_loss(alpha, size, beta):
        if size == 1:
            return beta * alpha
        elif size == 0:
            return 0
        return (np.log((size - 1)/size + np.exp((alpha-1) * 4 * beta * alpha) / size ) 
                / (alpha-1))
    @staticmethod 
    def noisy_privacy_loss(alpha, size, lambd, sigma):
        return (lambd / size)**2 * alpha / sigma**2 

    @staticmethod
    def data_indep_loss(alpha, size, beta=None, lambd=None, sigma=None):
        eps = 0 
        if beta is not None:
            eps += PMixED.sample_privacy_loss(alpha, size, beta)
        if lambd is not None:
            eps += PMixED.noisy_privacy_loss(alpha, size, lambd, sigma)
        return eps
    
    @staticmethod
    def data_dep_loss(mixed_dists, alpha, p_pub, ret_idx=False):
        max_loss = 0
        p = torch.stack(mixed_dists).mean(dim=0)
        mixed_dists = torch.stack(mixed_dists)
        for i in range(mixed_dists.size()[0]):
            p_i = torch.cat((mixed_dists[:i, :], mixed_dists[i+1:, :])).mean(dim=0)
            eps = PMixED.RDSym(p, p_i, alpha)
            max_loss = max(max_loss, eps)
        return max_loss
    

    @staticmethod
    def renyiDiv(p, q, alpha=float('inf')):
        if alpha == float('inf'):
            RD = torch.log(torch.max(p/q))
        elif alpha == 1:
            RD = torch.sum(p*torch.log(p/q))
        else:
            RD = 1/(alpha-1) * torch.log(
                torch.sum(((p/q)**(alpha))*q)
            )
        if torch.isnan(RD):
            RD = torch.log(torch.max(p/q))
        return RD 

    @staticmethod
    def RDSym(p_mix, p_pub, alpha):
        return  max(PMixED.renyiDiv(p_mix, p_pub, alpha=alpha),
                   PMixED.renyiDiv(p_pub, p_mix, alpha=alpha)).item()

    def lambda_solver_bisection(self, p, p_pub):
        def f(lambd):
            pred = self.mix(p, p_pub, lambd)
            eps = PMixED.RDSym(pred, p_pub, self.alpha)
            return (eps - self.target)
        if f(1) <= 0.0:
            lambd = 1 
        else:
            lambd = bisect(f, 0, 1, maxiter=20, disp=False)
        print(lambd)
        return lambd
    
    def beta_solver_bisection(self, size):
        def f(beta):
            sub_eps = self.subsample_eps(self.alpha, self.p, size=size, beta=beta)
            return (sub_eps - self.eps/self.q_budget)
        if f(80)<0:
            return 80
        else:
            return bisect(f, 0, 80, maxiter=20, disp=False)
            

    def poisson_subsample(self, number, p):
        '''
        Poisson Subsampling is equivalent to m ~ Binomial(n, p), 
        then sampling m without replacement from [N]
        '''
        return np.random.choice(number, np.random.binomial(number, p))

    def lambda_eq(self, p, p_pub):
        #x = min(self.beta * self.alpha * (self.alpha-1) -
        #         (self.alpha-1) * PMixED.RDSym(p, p_pub, self.alpha),
        #         0)
        #return np.exp(x)
        return min(
            (np.exp(self.beta * self.alpha * (self.alpha-1)) - 1) / 
            (np.exp((self.alpha-1) * PMixED.RDSym(p, p_pub, self.alpha)) - 1),
            1.0
        )

    def noisy_screen(self, pub_dist, priv_dists):
        '''
        Noisy Screening Mechanism:
        (1) Grab top-k entries V from public output distribution
        (2) Grab V entries from mixed distribution
        (3) Add noise to truncated mixed distribution
        (4) Renormalize probability distribution
        (5) Renyi Div between trunc mix dist and pub dist
        (6) if (4) > T then return pub dist else return mix dist
        (7) Record self.priv_loss 
        '''
        pub_logits = torch.log(pub_dist).view(1, -1)
        pub_logits, idxs_remov = self.top_k_filtering(pub_logits, self.screen_top_k)

        mix_dists = torch.stack([self.mix(priv_dist, pub_dist, self.lambd)
                                for priv_dist in priv_dists]).mean(dim=0)
        priv_logits = torch.log(mix_dists).view(1, -1)
        priv_logits[idxs_remov] = -float("Inf")

        trunc_pub_output_dist = F.softmax(pub_logits[-1, :], dim=-1)
        trunc_priv_output_dist = F.softmax(priv_logits[-1, :], dim=-1)

        idx_noise = torch.nonzero(trunc_priv_output_dist, as_tuple=True)[0]
        noise = torch.normal(0, self.sigma, size=(len(idx_noise), ))
        trunc_priv_output_dist[idx_noise] += noise

        min_output_dist = torch.abs(torch.min(trunc_priv_output_dist[idx_noise]))
        trunc_priv_output_dist[idx_noise] = (trunc_priv_output_dist[idx_noise]
                                                + min_output_dist) / (
                                trunc_priv_output_dist[idx_noise] + min_output_dist).sum()
        idxs = torch.nonzero(trunc_priv_output_dist)  
        rd_noisy = PMixED.RDSym(trunc_priv_output_dist[idxs],
                                  trunc_pub_output_dist[idxs],
                                  self.alpha)      
        return rd_noisy, noise

    def update_privacy_loss(self, sample=False, mixed_dists=None, p_pub=None, **kwargs):
        '''
        TODO: If noisy screening is not used, then don't account for it 
        '''
        loss = 0

        if (not sample) and self.accounting_method == "Independent":
            loss = self.subsample_eps(**kwargs) 
        elif (not sample) and self.accounting_method == "Dependent":
            loss = PMixED.noisy_privacy_loss(self.alpha,
                                             size=kwargs['size'],
                                             lambd=kwargs['lambd'],
                                             sigma=kwargs['sigma'])
        elif sample and self.accounting_method == "Independent":
            loss = self.subsample_eps(**kwargs)
        elif sample and self.accounting_method == "Dependent":
            data_dep_loss = PMixED.data_dep_loss(tuple(mixed_dists), kwargs['alpha'], p_pub)
            data_indep_loss = PMixED.sample_privacy_loss(self.alpha,
                                                     size=kwargs['size'],
                                                     beta=kwargs['beta'])
            loss = min(data_dep_loss, data_indep_loss) + PMixED.noisy_privacy_loss(self.alpha,
                                                                                   size=kwargs['size'],
                                                                                   lambd=kwargs['lambd'],
                                                                                   sigma=kwargs['sigma'])
        self.priv_loss += loss


    def gen_output_dist(self, question_ids, split_context, text,temperature):
        priv_dists = []
        number=len(split_context)-1
        if self.accounting_method != "Dependent":
            sampled = self.poisson_subsample(number,self.p)
        else:
            sampled = list(range(number))
        
        for i in sampled:
            context = torch.cat([question_ids, split_context[i], text], dim=0).unsqueeze(0)
            decoded_context = self.tokenizer.decode(context.squeeze())
            #print(decoded_context)
            context = context.to(self.device)
            logits = self.pub_model(context).logits.squeeze()
            if temperature < 1.0:
                logits = logits / temperature
            priv_dists.append(F.softmax(logits, dim=-1))
        context = torch.cat([question_ids, split_context[-1], text], dim=0).unsqueeze(0)
        context = context.to(self.device)
        decoded_context = self.tokenizer.decode(context.squeeze())
        #print(decoded_context)
        logits = self.pub_model(context).logits.squeeze()
        if temperature < 1.0:
            logits = logits / temperature
        pub_dist = F.softmax(logits, dim=-1)
        
        return pub_dist, priv_dists

    def gen_priv_output_dist(self, pub_dist, priv_dists):
        if len(priv_dists) == 0:
            self.num_non_sample += 1
            return pub_dist 

        if self.sigma is None:
            self.beta = self.beta_solver_bisection(len(priv_dists))
            self.target = self.beta * self.alpha

        self.lambdas = np.array([self.lambda_solver_bisection(priv_dist, pub_dist) for priv_dist in priv_dists])
        
        #self.lambdas = np.array([self.lambda_eq(priv_dists[i], pub_dist) for i in sampled])
        self.lambda_history.append(np.mean([lambd for lambd in self.lambdas]))
        mixed_dists = [self.mix(priv_dist, pub_dist, self.lambdas[lamb_i])
                       for lamb_i, priv_dist in enumerate(priv_dists)]
        #self.update_privacy_loss(sample=True,
        #                         alpha=self.alpha,
        #                         p=self.p,
        #                         size=len(sampled),
        #                         beta=self.beta,
        #                         lambd=self.lambd,
        #                         sigma=self.sigma,
        #                         mixed_dists=mixed_dists,
        #                         p_pub=pub_dist)
        output_dist = torch.stack(mixed_dists).mean(dim=0)
        #ls = self.fast_calc_ls(mixed_dists)
        #self.ls_cum += ls
        #self.ss = self.calc_ss(self.tau, self.ls_cum)
        return output_dist

    def mix(self, p, p_prime, lambd=0.5):
        return (lambd * p + (1-lambd) * p_prime)

    def fast_calc_ls(self, mixed_dists):
        max_ls = []
        N = len(mixed_dists)
        #avg_dists = torch.stack(mixed_dists).mean(dim=0)
        #dist_eps_pair = [(dist, PMixED.RDSym(avg_dists, dist, self.alpha)) for dist in mixed_dists]
        #dist_eps_pair.sort(key=lambda t: t[1])#, reverse=True)
        dists = torch.stack(mixed_dists)

        for d in range(0, N-2):
            avg_dists = dists.mean(dim=0)
            #dist_eps_pair = [(dist, PMixED.RDSym(avg_dists, dist, self.alpha)) for dist in dists]
            dist_eps_pair = [(dists[i], PMixED.RDSym(avg_dists,
                                                 torch.cat((dists[:i, :], 
                                                            dists[i+1:, :])).mean(dim=0),
                                                self.alpha))
                                                for i in range(N-d)]

            dist_eps_pair.sort(key=lambda t: t[1])#, reverse=True)
            #eps = PMixED.data_dep_loss(tuple(dists), self.alpha)
            eps = dist_eps_pair[-1][1]
            #dist_prime = [val[0] for val in dist_eps_pair[:N-d-1]]
            dist_prime = [val[0] for val in dist_eps_pair[:-1]]
            eps_prime = PMixED.data_dep_loss(tuple(dist_prime), self.alpha)
            ls = np.abs(eps-eps_prime)
            max_ls.append(ls)

            dists = torch.stack([val[0] for val in dist_eps_pair[1:]])
        return np.array(max_ls)

    def calc_ls(self, mixed_dists):
        N = len(mixed_dists)
        dist = mixed_dists
        max_ls = [] 
        for d in range(0, N-2):
            n = len(dist)
            dists_prime = combinations(dist, n-1)
            eps = PMixED.data_dep_loss(tuple(dist), self.alpha) 
            vals = [(dist_prime, PMixED.data_dep_loss(tuple(dist_prime), self.alpha)) for dist_prime in dists_prime]
            eps_prime = min(vals, key = lambda t: t[1])[1]
            dist = max(vals, key=lambda t: t[1])[0]
            ls = (eps-eps_prime)**2
            max_ls.append(ls)
        return np.array(max_ls)
    
    def calc_ss(self, tau, ls_cum):
        n = len(ls_cum)
        factors = np.exp(-tau * np.arange(n))
        return np.max(factors * ls_cum)
    

    
    def priv_generate(self, context, gram_length, question, min_length, max_length, temperature=1.0, top_k=None, tokenwise=True, endtoend=True, split=1):
        question="Here is the query: "+question
        split_context = generate_ngrams(context, self.tokenizer, self.device, tokenwise=tokenwise, endtoend=endtoend, split=split,n=gram_length)
        question_ids = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)["input_ids"].squeeze(0).to(self.device)
        stop_tokens_id = set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 
                          self.tokenizer.encode("\n")[0]])
        generated_token_ids = torch.empty(0, dtype=torch.long, device=self.device)
        number=len(split_context)-1

        self.p=self.Ex_num/number

        if(self.p>=0.99):
            self.p=0.99
        for i in range(max_length):
            with torch.no_grad():
                pub_dist, priv_dists = self.gen_output_dist(question_ids, split_context, generated_token_ids,temperature) 
                pub_dist = pub_dist[-1, :]
                if priv_dists:
                    priv_dists = [priv_dist[-1, :] for priv_dist in priv_dists]

                    if top_k is not None:
                        pub_logits = torch.log(pub_dist).view(1, -1)
                        pub_logits, idxs_remov = self.top_k_filtering(pub_logits, top_k)
                        
                        priv_logits = []
                        for priv_dist in priv_dists:
                            priv_logit = torch.log(priv_dist).view(1, -1)
                            priv_logit[idxs_remov] = -float("Inf")
                            priv_logits.append(priv_logit)

                        pub_dist = F.softmax(pub_logits, dim=-1).squeeze(0)
                        priv_dists = [F.softmax(priv_logit, dim=-1).squeeze(0) for priv_logit in priv_logits]
            priv_output_dist = self.gen_priv_output_dist(pub_dist, priv_dists)
            if i < min_length:
                for stop_token_id in stop_tokens_id:
                    priv_output_dist[stop_token_id] = 0
                priv_output_dist = priv_output_dist / priv_output_dist.sum() 
            next_token_id = priv_output_dist.multinomial(1).long().to(self.device)
            #print(f'Generated token "{self.tokenizer.decode(next_token_id)}"')
            if next_token_id.cpu().item() in stop_tokens_id:
                break

            
            generated_token_ids = torch.cat([generated_token_ids, next_token_id], dim=-1)

        return generated_token_ids

    @staticmethod 
    def top_k_filtering(logits, top_k, filter_value=-float("Inf")):
        #  Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits, indices_to_remove

    def print_priv_losses(self):
        print("-----------------------------")
        print(f"Total privacy loss {sum(self.priv_loss):.3f}")
        print("-----------------------------")
    
    def print_lambdas(self):
        print("-----------------------------")
        print(f"Average lambda value: {np.mean(self.lambda_history):.3f}")
        print("-----------------------------")
        print(f"Std lambda value: {np.std(self.lambda_history):.3f}")
        print("-----------------------------")
        print(f"Min lambda value: {np.min(self.lambda_history)}")
        print("-----------------------------")
        print(f"Max lambda value: {np.max(self.lambda_history):.3f}")
        print("-----------------------------")


    def plot_lambdas(self):
        x = [i for i in range(len(self.lambda_history))]
        plt.figure().set_figwidth(15)
        plt.plot(x, self.lambda_history)
        plt.grid()
        plt.savefig(os.path.join("plts", "lambda_vals.png"))
        plt.clf()
    
    @staticmethod
    def convert_to_aprox_dp(priv_loss, delta, alpha):
        return priv_loss + np.log((alpha-1)/alpha) - (np.log(delta) + np.log(alpha))/(alpha-1)
        
    def top_p_filtering(self, logits, top_p: float = 0.9, filter_value: float = -float("Inf")):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits, indices_to_remove