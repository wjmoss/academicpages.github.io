---
title: 'Double Machine Learning'
date: 2022-04-18
permalink: /posts/2022/04/dml/
tags:
  - double machine learning
  - treatment effect
  - theory

---

This blog post is some reading notes of the paper [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://arxiv.org/abs/1608.00060v6){:target="_blank"} by V. Chernozhukov et al.

Double Machine Learning is a framework to estimate a low-dimensional parameter of interest $\theta_0$, which is typically a causal parameter, in the presence of a high-dimensional nuisance parameter $\eta_0$. The nuisance parameter will be estimated using machine learning (ML), and the framework will give a root-$N$ consistent estimation of the parameter of interestunder regularity conditions.

1.Motivation
======
First let's focus on the example of a partially linear regression (PLR) model, where $Y$ is the outcome variable, $D$ is the policy/treatment variable of interest, $p$-dimensional vector $X$ are other controls, and $U$ and $V$ are disturbances.

\begin{align}
Y=D \theta_{0}+g_{0}(X)+U, & \mathrm{E}[U \mid X, D]=0 \\
D=m_{0}(X)+V, & \mathrm{E}[V \mid X]=0
\end{align}

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

In high-dimensional settings, regularization is necessary. It keep the variance from exploding but also induces substantive biases. Since $m_0(X)$ is not centered, the biases do not cancel out and term $b$ is the sum of $n$ terms that do not have zero mean. Specifically, the convergence rate of the bias of $\hat{g}_0$ is typically strictly slower than root-$N$ (i.e. bias $\sim O(n^{\psi_g})$ with $\psi_g<1/2$), hence the stochastic order of $b$ goes to infinity.

To avoid the retularization biases, we can first partial out the effect of $X$ on $D$ and then do the same procedures as above. That is, we obtain both $\hat{V}=D-\hat{m}_0(X)$ and $\hat{g}_0$ with auxiliary samples. Predicting $D$ with $X$ and predicting $g_0$ with $X$, this is the reason why the authors call the framework "double prediction" or "double machine learning".

Under this new construction, the estimator $\check{\theta}_0$ takes the form

$$
\check{\theta}_{0}=\left(\frac{1}{n} \sum_{i \in I} \widehat{V}_{i} D_{i}\right)^{-1} \frac{1}{n} \sum_{i \in I} \widehat{V}_{i}\left(Y_{i}-\widehat{g}_{0}\left(X_{i}\right)\right).
$$

and with the scaled error

$$
\sqrt{n}\left(\check{\theta}_{0}-\theta_{0}\right)=\underbrace{(E[V^2])^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} V_{i} U_{i}}_{:=a}+\underbrace{(E[V^2])^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} (\hat{m}_0(X_i)-m_0(X_i))\left(\hat{g}_{0}\left(X_{i}\right)-g_{0}\left(X_{i}\right)\right)}_{:=b}+\underbrace{\frac{1}{\sqrt{n}} \sum_{i \in I} V_{i} (\hat{g}_{0}(X_i)-g_0(X_i))}_{:=c}.
$$

The leading term $a^{\ast}$ again converges to a normal random variable under milde conditions. The second term $b^{\ast}$ is now upper-bounded by $\sqrt{n}n^{-(\psi_m+\psi_g)}$, which can vanish even though the rates $n^{-\psi_m}$ and $n^{-\psi_g}$ are slow. The sample splitting procedure (auxiliary part and main part of samples) ensures that $c^{\ast}=o_P(1)$ under weak conditions. Too see this, conditioning on the auxiliary sample, utilizing independence, and recalling that $E[V_i\mid X_i]=0$, it is easy to verify that $c^{\ast}$ has mean zero and variance of order

$$
\frac{1}{n} \sum_{i \in I}\left(\widehat{g}_{0}\left(X_{i}\right)-g_{0}\left(X_{i}\right)\right)^{2} \rightarrow_{P} 0.
$$

It looks like with sample splitting, we only use half of the samples to estimate the parameter of interest, and it can result in a  substantial loss of efficiency. However, we can simply filp the role of the main and auxiliary part to obtain another version of the estimator and average, or minimize the total loss combining two partial regression on $\theta_0$. In section 3 we will discuss the extension to a K-fold version of cross-fitting.


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

3.DML estimator and its properties
======
**DML 1** (averaging estimators)
1) Take a K-fold random partition $(I_{k})_{k=1}^{K}$ of observation indices $[N] =\{1,...,N\}$ such that the size of each fold $I_k$ is $n=N/K$. Also, for each $k \in[K]=\{1, ..., K\}$ define $I_{k}^{c}:=\{1, ..., N\}\backslash I_{k}$.
2) For each $k \in [K]$, construct a ML estimator
$$
\hat{\eta}_{0, k}=\hat{\eta}_{0}((W_{i})_{i \in I_{k}^{c}})
$$
of $\eta_0$ with all data part except $I_k$.
3) For each $k \in [K]$, construct the estimator $\check{\theta}_{0,k}$ as the
solution of the following equation:
$$
\mathbb{E}_{n, k}[\psi(W ; \check{\theta}_{0, k}, \hat{\eta}_{0, k})]=0,
$$
where $\psi$  is the Neyman orthogonal score, and $$\mathbb{E}_{n, k}[\psi(W)]=n^{-1} \sum_{i \in I_{k}} \psi\left(W_{i}\right)$$.
4) Aggregate the estimators:
$$
\tilde{\theta}_{0}=\frac{1}{K} \sum_{k=1}^{K} \check{\theta}_{0, k}.
$$

**DML 2** (combining all equations)
1) Take a K-fold random partition $(I_{k})_{k=1}^{K}$ of observation indices $[N] =\{1,...,N\}$ such that the size of each fold $I_k$ is $n=N/K$. Also, for each $k \in[K]=\{1, ..., K\}$ define $I_{k}^{c}:=\{1, ..., N\}\backslash I_{k}$.
2) For each $k \in [K]$, construct a ML estimator
$$
\hat{\eta}_{0, k}=\hat{\eta}_{0}((W_{i})_{i \in I_{k}^{c}})
$$
3) Consturct the estimator $\tilde{\theta}_0$ as the solution to
$$
\frac{1}{K} \sum_{k=1}^{K} \mathbb{E}_{n, k}\left[\psi\left(W ; \tilde{\theta}_{0}, \widehat{\eta}_{0, k}\right)\right]=0,
$$
$\psi$ and $\mathbb{E}_{n,k}$ are the same as above.

**Remark.**
The choice of $K$ has no asymptotic impact but may matter in small samples. The authors claim that moderate values of $K$ such as $4$ or $5$ work better than $K=2$ in empirical examples and simulations. They also recommend DML2 over DML1, because in most models (maybe except those with score function with $c\theta$ term, like ATE and ATTE?) the pooled empirical Jacobian for DML2 exhibits more stable behavior than the separate empirical Jacobians for DML1.

4.Applications
======




<!--References>
------

[https://www.cnblogs.com/gogoSandy/p/11711918.html](https://www.cnblogs.com/gogoSandy/p/11711918.html){:target="_blank"}
[https://zhuanlan.zhihu.com/p/115223013](https://zhuanlan.zhihu.com/p/115223013){:target="_blank"}

<!-- Aren't headings cool?
<!------>
