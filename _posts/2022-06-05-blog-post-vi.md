---
title: 'Variational Inference'
date: 2022-06-04
permalink: /posts/2022/06/nf/
tags: 
  - machine learning
  - EM algorithm
  - variational inference

---

1.Latent variable models and EM algorithm
======
We want to model a high dimension random vector $\boldsymbol{x}$, which can often be described by only a few latent factors $\boldsymbol{z}$. LVM take a two-step generation scheme:

1. Generate the latent variable $$\boldsymbol{z} \sim p_{\boldsymbol{\theta}}(\boldsymbol{z})$$
2. Generate the observed data $$\boldsymbol{x} \sim p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$$

The scheme defines the joint distribution 

$$
L(\theta,\boldsymbol{x},\boldsymbol{z}):=p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})=p_{\boldsymbol{\theta}}(\boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})
$$

and the marginal likelihood $$L(\theta,\boldsymbol{x}):=p_{\boldsymbol{\theta}}(\boldsymbol{x})=\mathbb{E}_{\boldsymbol{z}\sim p_\theta(\boldsymbol{z})}[p_\theta(\boldsymbol{x}\mid\boldsymbol{z})]$$

To find the MLE of the marginal likelihood, a classic method is the EM algorithm, which iteratively update the expectation of marginal likelihood $$Q(\theta\mid\theta^{(t)})=\mathbb{E}_{\boldsymbol{z}\mid \boldsymbol{x},\theta^{(t)}}\log L(\theta,\boldsymbol{x},\boldsymbol{z})$$
and the MLE $$\theta^{(t+1)}=\operatorname{arg}\max\limits_{\theta}Q(\theta\mid\theta^{(t)})$$.

proof of convergence:

Define

$$
H(\theta\mid\theta^{(t)}):=-\int p_{\theta^{(t)}}(\boldsymbol{z}\mid\boldsymbol{x})\log p_\theta(\boldsymbol{z}\mid\boldsymbol{x}) d\boldsymbol{z}
$$

$$
\begin{aligned}
\log p_\theta(\boldsymbol{x})&=\mathbb{E}_{\boldsymbol{z}\mid \boldsymbol{x},\theta^{(t)}}[\log p_\theta(\boldsymbol{x})]\\
&=\mathbb{E}_{\boldsymbol{z}\mid \boldsymbol{x},\theta^{(t)}}[\log p_\theta(\boldsymbol{x},\boldsymbol{z})-\log p_\theta(\boldsymbol{z}\mid \boldsymbol{x})]\\
&=Q(\theta\mid\theta^{(t)}) + H(\theta\mid\theta^{(t)})
\end{aligned}
$$

Since 

$$H(\theta\mid\theta^{(t)}) - H(\theta^{(t)}\mid\theta^{(t)})=\int \log \frac{p_{\theta^{(t)}}(\boldsymbol{z}\mid \boldsymbol{x})}{p_{\theta}({\boldsymbol{z}}\mid \boldsymbol{x})}p_{\theta^{(t)}}({\boldsymbol{z}}\mid \boldsymbol{x})dz=D_{KL}(p_{\theta^{(t)}}\vert\vert p_\theta)\geq 0,$$

we have $$\log p_{\theta^{(t+1)}}(\boldsymbol{x})-\log p_{\theta^{(t)}}(\boldsymbol{x})=Q(\theta^{(t+1)}\mid\theta^{(t)})-Q(\theta^{(t)}\mid\theta^{(t)})+D_{KL}(p_{\theta^{(t)}}\vert\vert p_{\theta^{(t+1)}})\geq 0,$$
which means the marginal likelihood increases at each iteration.

2.Variational inference: posterior distribution intractable
======
The idea of Variational Inference is very similar to EM algorithm. It also use the expectation over log, but with respect to an arbitrary distribution $q(\boldsymbol{z})$.

$$
\begin{aligned}
\log p_\theta(\boldsymbol{x})&=\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}[\log p_\theta(\boldsymbol{x})]\\
&=\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}[\log p_\theta(\boldsymbol{x},\boldsymbol{z})-\log p_\theta(\boldsymbol{z}\mid \boldsymbol{x})]\\
&=\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}\left[\log \frac{p_\theta(\boldsymbol{x},\boldsymbol{z})}{q(\boldsymbol{z})}+\log\frac{q(\boldsymbol{z})}{ p_\theta(\boldsymbol{z}\mid \boldsymbol{x})}\right]\\
&=\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}\left[\log \frac{p_\theta(\boldsymbol{x},\boldsymbol{z})}{q(\boldsymbol{z})}\right]+D_{KL}(q(\boldsymbol{z})\vert\vert p_\theta(\boldsymbol{z}\mid\boldsymbol{x}))\\
&:= Q(\theta\mid\theta^{(q)}) + H(\theta^{q}\mid\theta^{(q)}) + H(\theta\mid\theta^{(q)}) - H(\theta^{q}\mid\theta^{(q)})
\end{aligned}
$$

The first term

$$
L(\theta, q)=\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}\left[\log {p_\theta(\boldsymbol{x},\boldsymbol{z})}-\log{q(\boldsymbol{z})}\right]
$$

is called the Evidence Lower BOund (ELBO) of $\log p_\theta(\boldsymbol{x})$. The tightness of the bound depends on how close $a(\boldsymbol{z})$ is to the posterior $p_\theta(\boldsymbol{z}\mid\boldsymbol{x})$ in terms of KL divergence.

We can write the formula in another way

$$
L(\theta,q)=-D_{KL}(q(\boldsymbol{z})\vert\vert p_\theta(\boldsymbol{z}\mid\boldsymbol{x}))+\log p_\theta(\boldsymbol{x}).
$$

If we fix $\theta$, maximizing the ELBO (or even marginal likelihood) w.r.t. $q$ is equivalent to
making $q$ as close as possible to $p_\theta(\boldsymbol{z}\mid\boldsymbol{x})$ in terms of KL divergence. For fixed $q$, maximizing the ELBO  w.r.t. $\theta$ is equivalent to find $$\operatorname{arg}\max\limits_\theta\mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}[\log p_\theta(\boldsymbol{x},\boldsymbol{z})]$$.

The true posterior $p_\theta(\boldsymbol{z}\mid\boldsymbol{x})$ is often also intractable, so we cannot just set $q(\boldsymbol{z})=p_\theta(\boldsymbol{z}\mid\boldsymbol{x})$. When the posterior can be computed exactly, Variational Inference is the same as EM algorithm, which is just doing alternating optimization of the ELBO in a model.

3.Optimizing ELBO
======
We consider a parameterized family $\mathcal{Q}=\{q_\phi(\boldsymbol{z}),\phi\in\mathbb{R}^K\}$. The optimization of ELBO becomes

$$
\max\limits_{\theta\in\mathbb{R}^M,\phi\in\mathbb{R}^K}\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[\log p_\theta(\boldsymbol{x},\boldsymbol{z})-\log q_\phi(\boldsymbol{z})]=:\max\limits_{\theta,\phi}L(\theta,\phi).
$$

The standard technique is gradient ascent. We will not discuss second order methods here, so what we need is only the two gradients $\nabla_\theta L$ and $\nabla_\phi L$. In general the ELBO (as a integral) cannot be computed analytically, and we have to estimate the gradients from samples.

3.1 Computation of $\nabla_\theta$
------
Under mild regularity conditions (partial derivative controlled by an integral function $g$), the integral operator and the derivative operator are exchangable: 

$$
\nabla_\theta\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[f_\theta(\boldsymbol{z})]=\nabla_\theta\int q_\phi(\boldsymbol{z})f_\theta(\boldsymbol{z})d\boldsymbol{z}=\int q_\phi(\boldsymbol{z})\nabla_\theta f_\theta(\boldsymbol{z})d\boldsymbol{z}=\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[\nabla_\theta f_\theta(\boldsymbol{z})],
$$

for $f_\theta(\boldsymbol{z})=\log p_\theta(\boldsymbol{x},\boldsymbol{z})$. The right hand side can be approximated via Monte Carlo:

$$
\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[\nabla_\theta f_\theta(\boldsymbol{z})]\approx \frac{1}{S}\sum^S_{i=1}\nabla_\theta p_\theta(\boldsymbol{x}_i,\boldsymbol{z}_i).
$$

3.2  Computation of $\nabla_\phi$
------
We write $h_\phi(\boldsymbol{z})=\log p_\theta(\boldsymbol{x},\boldsymbol{z})-\log q_\phi(\boldsymbol{z})$. First we try the same way as that in 3.1, 

$$
\begin{aligned}
\nabla_\phi\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[h_\phi(\boldsymbol{z})]&=\nabla_\phi\int q_\phi(\boldsymbol{z})h_\phi(\boldsymbol{z})d\boldsymbol{z}=\int \nabla_\phi(q_\phi(\boldsymbol{z}) h_\phi(\boldsymbol{z}))d\boldsymbol{z}\\
&=\int(\nabla_\phi h_\phi(\boldsymbol{z})+h_\phi(\boldsymbol{z})\nabla_\phi\log q_\phi(\boldsymbol{z}))q_\phi(\boldsymbol{z})d\boldsymbol{z}.
\end{aligned}
$$

Ok, a bit weird. The standard idea is using reparameterization trick (will it lead to smaller variance?). Let $q_\phi(\boldsymbol{z})$ be a distribution that can be represented as a deterministic transformation $T(\epsilon, \phi)$ of some base distribution $b(\epsilon)$. For example, $q_\phi(\boldsymbol{z})=\mathcal{N}(\boldsymbol{z}\mid\boldsymbol{\mu},\boldsymbol{R}\boldsymbol{R}^T)$ can be constructed from $\epsilon\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ and $\boldsymbol{z}=T(\epsilon, \phi=(\boldsymbol{\mu},\boldsymbol{R}))=\boldsymbol{R}\epsilon+\boldsymbol{\mu}$.

Under the reparameterization, we can perform exactly the same exchange of integral and derivative.

$$
\begin{aligned}
\nabla_\phi\mathbb{E}_{\boldsymbol{z}\sim q_\phi(\boldsymbol{z})}[h_\phi(\boldsymbol{z})]
&=\nabla_\phi\int q_\phi(\boldsymbol{z})h_\phi(\boldsymbol{z})d\boldsymbol{z}
=\nabla_\phi\int b(\epsilon) h_\phi(T(\epsilon,\phi))d\epsilon\\
&=\int b(\epsilon)\nabla_\phi h_\phi(T(\epsilon,\phi))d\epsilon\\
&=\mathbb{E}_{\epsilon\sim b(\epsilon)}[\nabla_\phi h_\phi(T(\epsilon,\phi))]\\
&\approx \frac{1}{S}\sum^S_{i=1}\nabla_\phi h_\phi(T(\epsilon_i,\phi))
\end{aligned}
$$

3.3 Mean Field Assumption
------
The general latent distribution $q(\boldsymbol{Z})$ allows possible dependencies between the latent variables $\boldsymbol{z}_i$ for different data points $i$. In practice it is often assumed that $q(\boldsymbol{Z})$ factorizes:

$$
q(\boldsymbol{Z})=\prod^N_{i=1}q_i(\boldsymbol{z}_i).
$$

This assumption is called the Mean Field Assumption. 

Under this assumption, it is easier to model the distribution $q(\boldsymbol{z}_i)$ in $\mathbb{R}^L$ rather than $q(\boldsymbol{Z})$ in $\mathbb{R}^{L\times N}$. And the ELBO can be decomposed into the sum of $N$ terms with parameter $(\theta, q_i)$.

The Mean Field Assumption simplifies the computation, but also restricts the expression capability. The dependencies / correlations across samples are not incorparated, which is not a problem with good sample assumptions (i.i.d., exchangable?) in classic statistics setup. "If the data is i.i.d., this assumption is a often a pretty good approximation. However, If the true posterior is highly correlated, the approximation can be poor."


References
------

Machine Learning for Graphs and Sequential Data (IN2323), TUM S21 


<!-- Aren't headings cool?
<!------>
