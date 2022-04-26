---
title: 'Double Machine Learning'
date: 2022-04-18
permalink: /posts/2022/04/dml/
tags:
  - double machine learning
  - treatment effect
  - theory

---

This blog post is some reading notes of the paper [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://arxiv.org/abs/1608.00060v6){:target="_blank"} by Victor Chernozhukov et al.

Double Machine Learning is a framework to estimate a low-dimensional parameter of interest $\theta_0$, which is typically a causal parameter, in the presence of a high-dimensional nuisance parameter $\eta_0$. The nuisance parameter will be estimated using machine learning (ML), and the framework will give a root-$N$ consistent estimation of the parameter of interestunder regularity conditions.

<br/>
1.Motivation, the example of PLR
======
First let's focus on the example of a partially linear regression (PLR) model, where $Y$ is the outcome variable, $D$ is the policy/treatment variable of interest, $p$-dimensional vector $X$ are other controls, and $U$ and $V$ are disturbances.

$$
\begin{aligned}
Y=D \theta_{0}+g_{0}(X)+U, &\ \mathrm{E}[U \mid X, D]=0 \\
D=m_{0}(X)+V, &\ \mathrm{E}[V \mid X]=0
\end{aligned}
$$

A naive approach to estimate $\theta_0$ is to construct $\hat{g}_0$ by $n=N/2$ auxiliary samples (denonted by $I^c$) and then regress $\hat{\theta}_0$ with the other half main samples (denoted by $I$). Recalling the formula of bivariate regression, we have:

$$
\hat{\theta}_{0}=\left(\frac{1}{n} \sum_{i \in I} D_{i}^{2}\right)^{-1} \frac{1}{n} \sum_{i \in I} D_{i}\left(Y_{i}-\hat{g}_{0}\left(X_{i}\right)\right).
$$

The scaled estimation error can be written as

$$
\sqrt{n}\left(\hat{\theta}_{0}-\theta_{0}\right)=\underbrace{\left(\frac{1}{n} \sum_{i \in I} D_{i}^{2}\right)^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} D_{i} U_{i}}_{:=a}+\underbrace{\left(\frac{1}{n} \sum_{i \in I} D_{i}^{2}\right)^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} D_{i}\left(g_{0}\left(X_{i}\right)-\hat{g}_{0}\left(X_{i}\right)\right)}_{:=b}.
$$

Term $a$ converges to some zero-mean normal random variable under mild conditions. Term $b$ is the bias term, which is not centered and diverges in general, when $X$ is high dimensional or has other highly complex settings. Indeed, the second term is of the form

$$
b=\left(\mathrm{E} D_{i}^{2}\right)^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} m_{0}\left(X_{i}\right)\left(g_{0}\left(X_{i}\right)-\widehat{g}_{0}\left(X_{i}\right)\right)+o_{P}(1).
$$

In high-dimensional settings, regularization is necessary. It keep the variance from exploding but also induces substantive biases. Since $m_0(X)$ is not centered, the biases do not cancel out and term $b$ is the sum of $n$ terms that do not have zero mean. Specifically, the convergence rate of the bias of $\hat{g}_0$ is typically strictly slower than root-$N$ (i.e. bias $\sim O(n^{-\psi_g})$ with $\psi_g<1/2$), hence the stochastic order of $b$ goes to infinity.

To avoid the retularization biases, we can first partial out the effect of $X$ on $D$ and then do the same procedures as above. That is, we obtain both $\hat{V}=D-\hat{m}_0(X)$ and $\hat{g}_0$ with auxiliary samples. Predicting $D$ with $X$ and predicting $g_0$ with $X$, this is the reason why the authors call the framework "double prediction" or "double machine learning".

Under this new construction, the estimator $\check{\theta}_0$ takes the form

$$
\check{\theta}_{0}=\left(\frac{1}{n} \sum_{i \in I} \widehat{V}_{i} D_{i}\right)^{-1} \frac{1}{n} \sum_{i \in I} \widehat{V}_{i}\left(Y_{i}-\widehat{g}_{0}\left(X_{i}\right)\right).
$$

and with the scaled error

$$
\sqrt{n}\left(\check{\theta}_{0}-\theta_{0}\right)=\underbrace{(E[V^2])^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} V_{i} U_{i}}_{:=a^\ast}+\underbrace{(E[V^2])^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} (\hat{m}_0(X_i)-m_0(X_i))\left(\hat{g}_{0}\left(X_{i}\right)-g_{0}\left(X_{i}\right)\right)}_{:=b^\ast}+\underbrace{\frac{1}{\sqrt{n}} \sum_{i \in I} V_{i} (\hat{g}_{0}(X_i)-g_0(X_i))}_{:=c^\ast}.
$$

The leading term $a^{\ast}$ again converges to a normal random variable under milde conditions. The second term $b^{\ast}$ is now upper-bounded by $\sqrt{n}n^{-(\psi_m+\psi_g)}$, which can vanish even though the rates $n^{-\psi_m}$ and $n^{-\psi_g}$ are slow. The sample splitting procedure (auxiliary part and main part of samples) ensures that $c^{\ast}=o_P(1)$ under weak conditions. Too see this, conditioning on the auxiliary sample, utilizing independence, and recalling that $E[V_i\mid X_i]=0$, it is easy to verify that $c^{\ast}$ has mean zero and variance of order

$$
\frac{1}{n} \sum_{i \in I}\left(\widehat{g}_{0}\left(X_{i}\right)-g_{0}\left(X_{i}\right)\right)^{2} \rightarrow_{P} 0.
$$

It looks like with sample splitting, only half of the samples are used to estimate the parameter of interest, and it can result in a substantial loss of efficiency. However, the role of the main and auxiliary parts can be flipped to obtain another version of the estimator for averaging. Another idea is to minimize the total loss combining two partial regression on $\theta_0$. In section 3 the authors propose the extension to a K-fold version of cross-fitting.

<br/>
2.Neyman orthogonality and moment conditions
======
The first "conventional" estimator $\hat{\theta}_0$ can be viewed as a solution to a equation

$$
\frac{1}{n} \sum_{i \in I} \varphi\left(W ; \hat{\theta}_{0}, \hat{g}_{0}\right)=0.
$$

In the PLR model above, the score function is $$\varphi(W ; \theta, g)=(Y-\theta D-g(X)) D$$. This score function is sensitive to the biased estimation of $g$, indeed, the [Gateaux derivative](https://en.wikipedia.org/wiki/Gateaux_derivative){:target="_blank"} operator w.r.t $g$ (direction derivative at the direction $g$) does not vanish: $$\partial_{g} \mathrm{E} \varphi\left(W ; \theta_{0}, g_{0}\right)\left[g-g_{0}\right] \neq 0$$.

By contrast, the doubled ML esimator $\check{\theta}_0$ solves $$
\frac{1}{n} \sum_{i \in I} \psi\left(W ; \check{\theta}_{0}, \hat{\eta}_{0}\right)=0$$ with Gateaux derivative vanishing score function $\psi(W;\theta,\eta)=(Y-\theta D-g(X))(D-m(X))$: $$\partial_{\eta} \mathrm{E} \psi\left(W ; \theta_{0}, \eta_{0}\right)\left[\eta-\eta_{0}\right]=0$$. Notice that here the nuisance parameter $\eta_0=(g_0,m_0)$ and $\hat{\eta}_0$ is the estimator.

The property of Gateaux derivative vanishing is called Neyman orthogonality and the correpdoning $\psi$ is the Neyman orthogonal score function. The monent conditions used to identify $\theta_0$ are locally insensitive to the bias of estimation to nuisance parameter, which is the key to good behavior of the estimation $\check{\theta}_0$. 

Now we know there are some score function with desired property, but how can we construction this kind of score function in general? The authors then introduce a method for likelihood and other M-estimators, with finite dimensional nuisance parameters.

Let $\theta\in\Theta\subset\mathbb{R}^{d_\theta}$ and $\beta\in\mathcal{B}\subset\mathbb{R}^{d_\beta}$ ($\mathcal{B}$ convex), be the target and the nuisance parameters. Suppose that the true parameter values $\theta_0$ and $\beta_0$ solve the optimization problem

$$
\max _{\theta \in \Theta, \beta \in \mathcal{B}} \mathrm{E}_{P}[\ell(W ; \theta, \beta)],
$$

where $$\ell(W ; \theta, \beta)$$ is a known criterion function (refered as the quasi-log-likelihood function). Then the partial derivative of the optimization objective on $\theta_0$ and $\beta_0$ are zero, under mild regularity conditions. The Neyman orthogonal score function is constructed as

$$
\psi(W ; \theta, \eta)=\partial_{\theta} \ell(W ; \theta, \beta)-\mu \partial_{\beta} \ell(W ; \theta, \beta),
$$

where the nuisance parameter is $$\eta=\left(\beta^{\prime}, \operatorname{vec}(\mu)^{\prime}\right)^{\prime} \in T=\mathcal{B} \times \mathbb{R}^{d_{\theta} d_{\beta}} \subset \mathbb{R}^{p}, \quad p=d_{\beta}+d_{\theta} d_{\beta}$$, and $\mu$ is the $d_\theta\times d_\beta$ orthogonalization parameter matrix whose true value $\mu_0$ satisfies $$J_{\theta \beta}-\mu J_{\beta \beta}=0$$ for 

$$
J=\left(\begin{array}{cc}
J_{\theta \theta} & J_{\theta \beta} \\
J_{\beta \theta} & J_{\beta \beta}
\end{array}\right)=\left.\partial_{\left(\theta^{\prime}, \beta^{\prime}\right)} \mathrm{E}_{P}\left[\partial_{\left(\theta^{\prime}, \beta^{\prime}\right)^{\prime}} \ell(W ; \theta, \beta)\right]\right|_{\theta=\theta_{0} ; \beta=\beta_{0}} .
$$

<br/>
3.DML estimator and its properties
======

3.1 DML estimator, 2 verions
------

**DML 1** (averaging estimators)

1) Take a K-fold random partition $$(I_{k})_{k=1}^{K}$$ of observation indices $$[N] =\{1,...,N\}$$ such that the size of each fold $I_k$ is $n=N/K$. Also, for each $$k \in[K]=\{1, ..., K\}$$ define $$I_{k}^{c}:=\{1, ..., N\}\backslash I_{k}$$.

2) For each $k \in [K]$, construct a ML estimator
$$
\hat{\eta}_{0, k}=\hat{\eta}_{0}((W_{i})_{i \in I_{k}^{c}})
$$
of $\eta_0$ with all data part except $I_k$.

3) For each $k \in [K]$, construct the estimator $$\check{\theta}_{0,k}$$ as the
solution of the following equation:
$$
\mathbb{E}_{n, k}[\psi(W ; \check{\theta}_{0, k}, \hat{\eta}_{0, k})]=0,
$$
where $\psi$  is the Neyman orthogonal score, and $$\mathbb{E}_{n, k}[\psi(W)]=n^{-1} \sum_{i \in I_{k}} \psi\left(W_{i}\right)$$.

4) Aggregate the estimators:
$$
\tilde{\theta}_{0}=\frac{1}{K} \sum_{k=1}^{K} \check{\theta}_{0, k}.
$$

<br/>
**DML 2** (combining all equations)

1) Take a K-fold random partition $$(I_{k})_{k=1}^{K}$$ of observation indices $$[N] =\{1,...,N\}$$ such that the size of each fold $I_k$ is $n=N/K$. Also, for each $$k \in[K]=\{1, ..., K\}$$ define $$I_{k}^{c}:=\{1, ..., N\}\backslash I_{k}$$.

2) For each $k \in [K]$, construct a ML estimator
$$
\hat{\eta}_{0, k}=\hat{\eta}_{0}((W_{i})_{i \in I_{k}^{c}})
$$

3) Consturct the estimator $$\tilde{\theta}_0$$ as the solution to
$$
\frac{1}{K} \sum_{k=1}^{K} \mathbb{E}_{n, k}\left[\psi\left(W ; \tilde{\theta}_{0}, \widehat{\eta}_{0, k}\right)\right]=0,
$$
$\psi$ and $$\mathbb{E}_{n,k}$$ are the same as above.

<br/>
**Remark.**
The choice of $K$ has no asymptotic impact but may matter in small samples. The authors claim that moderate values of $K$ such as $4$ or $5$ work better than $K=2$ in empirical examples and simulations. They also recommend DML2 over DML1, because in most models (perhaps except those with score function with $c\cdot\theta$ term, like ATE and ATTE?) the pooled empirical Jacobian for DML2 exhibits more stable behavior than the separate empirical Jacobians for DML1.


3.2 Assumptions and properties
------
The theory part of this paper is very long and complicated. I just summarize results for models with linear scores, since all models described in the Application part have score function linear in $\theta$:

$$
\psi(w ; \theta, \eta)=\psi^{a}(w ; \eta) \theta+\psi^{b}(w ; \eta), \quad \text { for all } w \in \mathcal{W}, \theta \in \Theta, \eta \in T.
$$

<br/><br/>
Let $c_1\geq c_0>0$, $s>0$, and $q>2$ be finite constants. Let $$\{\delta_N\}_{N\geq 1}$$ and $$\{\Delta_N\}_{N\geq 1}$$ be some sequences of positive constants converging to zero and $$\delta_N\geq N^{-1/2}$$. Also, let $$K\geq 2$$ be a fixed integer, and let $$\{\mathcal{P}_N\}_{N\geq 1}$$ be a sequence of sets of probability distributions $P$ of $W$ on $\mathcal{W}$.

<br/>
**Assumption 3.1** (Linear scores with approximate Neyman orthogonality)

For all $N\geq 3$ and $$P\in\mathcal{P}_N$$, the following conditions hold.

a) The true parameter value $\theta_0$ satisfies $$E_P[\psi(W;\theta_0,\eta_0)]=0$$.

b) The score $\psi$ is linear in the sense of (12).

c) The map $$\eta \mapsto \mathrm{E}_{P}[\psi(W ; \theta, \eta)]$$ is twice continuously Gateaux-differentiable on $T$.

d) The score $\psi$ obeys the Neyman (near-)orthogonality condition at $(\theta_0, \eta_0)$ w.r.t. the nuisance realization set $\mathcal{T}_N\subset T$ for 
$$
\lambda_{N}:=\sup _{\eta \in \mathcal{T}_{N}}\left\|\partial_{\eta} \mathrm{E}_{P} \psi\left(W ; \theta_{0}, \eta_{0}\right)\left[\eta-\eta_{0}\right]\right\| \leqslant \delta_{N} N^{-1 / 2}
$$

e) The identification condition holds, namely, the singular values of the matrix $$J_{0}:=\mathrm{E}_{P}\left[\psi^{a}\left(W ; \eta_{0}\right)\right]$$ are between $c_0$ and $c_1$.

<br/>
**Assumption 3.2** (score retularity and quality of nuisance parameter estimators)

For all $N\geq 3$ and $$P\in\mathcal{P}_N$$, the following conditions hold.

a) Given a random subset $I$ of $[N]$ of size $n=N/K$, the nuisance parameter estimator $$\hat{\eta}_{0, k}=\hat{\eta}_{0}((W_{i})_{i \in I_{k}^{c}})$$ belongs to the $$\mathcal{T}_N$$ with probability at least $$1-\Delta_N$$, where $$\mathcal{T}_N$$ contains $\eta_0$ and is constrained by the next conditions.

b) The moment conditions hold:

$$
\begin{aligned}
m_{N} &:=\sup _{\eta \in \mathcal{T}_{N}}\left(\mathrm{E}_{P}\left[\left\|\psi\left(W ; \theta_{0}, \eta\right)\right\|^{q}\right]\right)^{1 / q} \leqslant c_{1} \\
m_{N}^{\prime} &:=\sup _{\eta \in \mathcal{T}_{N}}\left(\mathrm{E}_{P}\left[\left\|\psi^{a}(W ; \eta)\right\|^{q}\right]\right)^{1 / q} \leqslant c_{1}
\end{aligned}
$$

c) The conditions on the statistical rates $r_N, r'_N$ and $\lambda_N$ hold:

$$
\begin{aligned}
r_{N} &:=\sup _{\eta \in \mathcal{T}_{N}}\left\|\mathrm{E}_{P}\left[\psi^{a}(W ; \eta)\right]-\mathrm{E}_{P}\left[\psi^{a}\left(W ; \eta_{0}\right)\right]\right\| \leqslant \delta_{N}, \\
r_{N}^{\prime} &:=\sup _{\eta \in \mathcal{T}_{N}}\left(\mathrm{E}_{P}\left[\left\|\psi\left(W ; \theta_{0}, \eta\right)-\psi\left(W ; \theta_{0}, \eta_{0}\right)\right\|^{2}\right]\right)^{1 / 2} \leqslant \delta_{N}, \\
\lambda_{N}^{\prime} &:=\sup _{r \in(0,1), \eta \in \mathcal{T}_{N}}\left\|\partial_{r}^{2} \mathrm{E}_{P}\left[\psi\left(W ; \theta_{0}, \eta_{0}+r\left(\eta-\eta_{0}\right)\right)\right]\right\| \leqslant \delta_{N} / \sqrt{N} .
\end{aligned}
$$

d) The variance of the score $\psi$ is non-degenerate: all eigenvalues of the matrix $$\mathrm{E}_{P}\left[\psi\left(W ; \theta_{0}, \eta_{0}\right) \psi\left(W ; \theta_{0}, \eta_{0}\right)^T\right]$$ are bounded from below by $c_0$.

<br/>
**Remark on Assumption 3.2**

When the map $$(\theta, \eta) \mapsto \psi(W ; \theta, \eta)$$ is smooth, the rates can be bounded

$$
\begin{aligned}
r_{N} \lesssim \varepsilon_{N}, \quad r_{N}^{\prime} \lesssim \varepsilon_{N}, \quad \lambda_{N}^{\prime} \lesssim \varepsilon_{N}^{2},
\end{aligned}
$$

where $\varepsilon_N$ is the upper bound on the rate of convergence of $\hat{\eta}_0$ to $\eta_0$ with respect to the $L_2$ norm under $P$.

If we assume that the necessary condition (?) holds, Assumption 3.2,
particularly $$\lambda_{N}^{\prime}=o\left(N^{-1 / 2}\right)$$, imposes the (crude) rate requirement

$$
\varepsilon_{N}=o\left(N^{-1 / 4}\right),
$$

which can be achieved by many machine learning methods under structured assumptions on the nuisance parameters.

<br/>

Suppose that Assumption 3.1 and 3.2 hold, these are the main theorems:

<br/>
**Theorem 3.1** (properties of the DML)

The DML1 and DML2 estimators $\tilde{\theta}_0$ concentrate in a $1/\sqrt{N}$ neighborhood of $\theta_0$ and are approximately linear and centred Gaussian,

$$
\sqrt{N} \sigma^{-1}\left(\tilde{\theta}_{0}-\theta_{0}\right)=\frac{1}{\sqrt{N}} \sum_{i=1}^{N} \bar{\psi}\left(W_{i}\right)+O_{P}\left(\rho_{N}\right) \leadsto N\left(0, \mathrm{I}_{d}\right)
$$

uniformly over $P\in\mathcal{P}_N$, and

$$
\rho_{N}:=N^{-1 / 2}+r_{N}+r_{N}'+N^{1 / 2} \lambda_{N}+N^{1 / 2} \lambda_{N}' \lesssim \delta_{N}
$$

Here, $$\bar{\psi}(\cdot):=-\sigma^{-1} J_{0}^{-1} \psi\left(\cdot, \theta_{0}, \eta_{0}\right)$$ is the [influence function](https://en.wikipedia.org/wiki/Robust_statistics#Influence_function_and_sensitivity_curve), and the approximate variance is

$$
\sigma^{2}:=J_{0}^{-1} \mathrm{E}_{P}\left[\psi\left(W ; \theta_{0}, \eta_{0}\right) \psi\left(W ; \theta_{0}, \eta_{0}\right)^{\prime}\right]\left(J_{0}^{-1}\right)^T.
$$

<br/>
**Theorem 3.2** (variance estimator for DML)

Consider the following estimator of the asymptotic variance matrix of $\sqrt{N}(\tilde{\theta}_0-\theta_0)$:

$$
\hat{\sigma}^{2}=\widehat{J}_{0}^{-1} \frac{1}{K} \sum_{k=1}^{K} \mathbb{E}_{n, k}\left[\psi\left(W ; \tilde{\theta}_{0}, \hat{\eta}_{0, k}\right) \psi\left(W ; \tilde{\theta}_{0}, \hat{\eta}_{0, k}\right)^{\prime}\right]\left(\hat{J}_{0}^{-1}\right)^T,
$$

where

$$
\widehat{J}_{0}=\frac{1}{K} \sum_{k=1}^{K} \mathbb{E}_{n, k}\left[\psi^{a}\left(W ; \hat{\eta}_{0, k}\right)\right]
$$

and $\tilde{\theta}_0$ is either the DML1 or DML2 estimator. This estimator satisfies

$$
\hat{\sigma}^{2}=\sigma^{2}+O_{P}\left(\varrho_{N}\right), \quad \varrho_{N}:=N^{-[(1-2 / q) \wedge 1 / 2]}+r_{N}+r_{N}'\lesssim \delta_{N} .
$$

Moreover, $\hat{\sigma}^2$ can replace $\sigma^2$ in the statement of Theorem 3.1 with the reminder rate updated as

$$
\rho_{N}=N^{-[(1-2 / q) \wedge 1 / 2]}+r_{N}+r_{N}'+N^{1 / 2} \lambda_{N}+N^{1 / 2} \lambda_{N}'.
$$

<br/><br/>
Based on Theorem 3.1 and 3.2, confidence intervals can be constructed. This part is omitted in this note. I also skip the subsection about non-linear scores.

<br/>

The the specfic sample partition has no impact on estimation results asymptotically, the effect of the particular random split on the estimate can be important in finite samples. In practice, different random split generally gives different estimates. To make the results more robust w.r.t. partitions, the authors propose to use mean or median of $S$ DML estimators.

The point estimators are given by 

$$
\tilde{\theta}_{0}^{\text {mean }}=\frac{1}{S} \sum_{s=1}^{S} \tilde{\theta}_{0}^{s} \quad \text { or } \quad \tilde{\theta}_{0}^{\text {median }}=\operatorname{median}\left\{\tilde{\theta}_{0}^{s}\right\}_{s=1}^{S},
$$

and the variance estimators are

$$
\hat{\sigma}^{2, \text { mean }}=\frac{1}{S} \sum_{s=1}^{S}\left(\hat{\sigma}_{s}^{2}+\left(\hat{\theta}_{s}-\tilde{\theta}^{\text {mean }}\right)\left(\widehat{\theta}_{s}-\tilde{\theta}^{\text {mean }}\right)^T\right)
$$

and

$$
\hat{\sigma}^{2, \text { median }}=\operatorname{median}\left\{\hat{\sigma}_{s}^{2}+\left(\left(\hat{\theta}_{s}-\tilde{\theta}^{\text {median }}\right)\left(\hat{\theta}_{s}-\tilde{\theta}^{\text {median }}\right)^T\right\}_{s=1}^{S}\right.
$$

If $S$ is fixed, the statements in Theorem 3.1 and 3.2 still hold providing Assumption 3.1 and 3.2, with $\hat{\theta}_0$ and $\hat{\sigma}$ replaced by the mean or median version.

The authors recommend using medians, as correponding quantities are more robust to outliers.

<br/>

4.Applications
======
tbd



References
------

[https://matheusfacure.github.io/python-causality-handbook/22-Debiased-Orthogonal-Machine-Learning.html](https://matheusfacure.github.io/python-causality-handbook/22-Debiased-Orthogonal-Machine-Learning.html){:target="_blank"}

[Double Machine Learning for causal inference](https://towardsdatascience.com/double-machine-learning-for-causal-inference-78e0c6111f9d){:target="_blank"}


<!-- Aren't headings cool?
<!------>
