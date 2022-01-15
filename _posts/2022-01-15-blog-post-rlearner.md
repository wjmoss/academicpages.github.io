---
title: 'R-learner'
date: 2022-01-15
permalink: /posts/2022/01/rlearner/
tags:
  - causal inference
  - treatment effect
  - theory

---


This blog post is some reading notes of the paper [Quasi-oracle estimation of heterogeneous treatment effects](https://academic.oup.com/biomet/article-abstract/108/2/299/5911092){:target="_blank"} by Xinkun Nie and Stefan Wager.

1.Introduction
======
The paper "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" constructs a novel framework for conditional average treatment effect estimation in observational studies. The framework has two steps: (1) estimating conditional mean outcome and propensity score; (2) estimating CATE by plugging in the estimates in (1). In each step, various machine learning method can be applied.

2.Problem setups
======
For each individual $i$, the triple $(X_i,Y_i,W_i)$ correspond to respectively per-person features, observed outcome and treatment assignment. The pair $(Y_i(0),Y_i(1))$ represent potential outcomes, and

$$
Y_{i}^{\mathrm{obs}}=Y_{i}\left(W_{i}\right)= \begin{cases}
Y_{i}(0) & \text { if } W_{i}=0, \\
Y_{i}(1) & \text { if } W_{i}=1.
\end{cases}
$$

The unit-level causal effect is $\tau_i=Y_i(1)-Y_i(0)$, and the conditional average treatment effect is defined by $\tau^\ast(x) := \mathbb{E}\left[Y_{i}(1)-Y_{i}(0) \mid X_{i}=x\right]$. Moreover, we need to define the treatment propensity score $e^{\ast}(x)=P(W=1 \mid X=x)$ and the conditional response surfaces $\mu_{(w)}^{\ast}(x)=E\{Y(w) \mid X=x\}$ for $w\in\{0,1\}$. 

Finally, the framework works under the unconfoundedness assumption.

**Assumption 1 (Unconfoundedness).** 
$$\left\{Y_{i}(0), Y_{i}(1)\right\} {\perp\kern-1.3ex\perp} W_{i} \mid X_{i}.$$


3.R-learner
======
Under unconfoundedness assumption, we can check that

$$
E\left\{\varepsilon_{i}\left(W_{i}\right) \mid X_{i}, W_{i}\right\}=0, \text{ where } \varepsilon_{i}(w):=Y_{i}(w)-\{\mu_{(0)}^{\ast}\left(X_{i}\right)+w \tau^{\ast}\left(X_{i}\right)\}.
$$

Robinson's transformation rewrites the equation with the propensity score $e^\ast(x)$, the CATE $\tau^\ast(x)$, and the conditional mean outcome $m^{\ast}(x):=E(Y \mid X=x)=\mu_{(0)}^{\ast}\left(X_{i}\right)+e^{\ast}\left(X_{i}\right) \tau^{\ast}\left(X_{i}\right)$.

$$
Y_{i}-m^{\ast}\left(X_{i}\right)=\left\{W_{i}-e^{\ast}\left(X_{i}\right)\right\} \tau^{\ast}\left(X_{i}\right)+\varepsilon_{i}.
$$

The CATE function can be expressed in the form of a minimizer:

$$
\tau^{\ast}(\cdot)=\operatorname{argmin}_{\tau}\left\{E\left(\left[\left\{Y_{i}-m^{\ast}\left(X_{i}\right)\right\}-\left\{W_{i}-e^{\ast}\left(X_{i}\right)\right\} \tau\left(X_{i}\right)\right]^{2}\right)\right\}.
$$

An oracle who knows both the functions $m^\ast(x)$ and $e^\ast(x)$ a priori could estimate the heterogeneous treatment effect function by empirical loss minimization,

$$
\tilde{\tau}(\cdot)=\operatorname{argmin}_{\tau}\left(\frac{1}{n} \sum_{i=1}^{n}\left[\left\{Y_{i}-m^{\ast}\left(X_{i}\right)\right\}-\left\{W_{i}-e^{\ast}\left(X_{i}\right)\right\} \tau\left(X_{i}\right)\right]^{2}+\Lambda_{n}\{\tau(\cdot)\}\right),\tag{5}
$$

but usually $m^\ast(x)$ and $e^\ast(x)$ are unknown and this estimator is not applicable.

The paper proposes the R-learner (Robinson's transformation type) framework of two-step estimators using cross-fitting:

1.
------

Divide up the data into $Q$ (typically set to 5 or 10) evenly sized folds. Let $q(\cdot)$ be a mapping from the $i = 1,\dots,n$ sample indices to $Q$ evenly sized data folds, fit $\hat{m}$ and $\hat{e}$ with cross-fitting over the $Q$ folds via methods tuned for optimal predictive accuracy.

2.
------

Estimate treatment effects via a plug-in version of $(5)$, where $\hat{e}^{(-q(i))}$ etc., denote predictions made without using the data fold the $i$-th training example belongs to,

$$
\hat{\tau}(\cdot)=\operatorname{argmin}_{\tau}\left[\widehat{L}_{n}\{\tau(\cdot)\}+\Lambda_{n}\{\tau(\cdot)\}\right], \\
\widehat{L}_{n}\{\tau(\cdot)\}=\frac{1}{n} \sum_{i=1}^{n}\left[\left\{Y_{i}-\hat{m}^{(-q(i))}\left(X_{i}\right)\right\}-\left\{W_{i}-\hat{e}^{(-q(i))}\left(X_{i}\right)\right\} \tau\left(X_{i}\right)\right]^{2}.
$$


4.Error rate
======
Let $\mathcal{P}$ be a non-negative measure over the compact metric space $\mathcal{X}\subseteq\mathbb{R}^d$. 

$$
T_{\mathcal{K}}(f)(\cdot)=\mathbb{E}\left[\mathcal{K}(\cdot, X) f(X)\right].
$$

Mercer's theorem implies that $T_{\mathcal{K}}$ has an orthogonal basis $\left(\psi_{j}\right)_{j=1}^{\infty}$ and corresponding eigen values $\left(\sigma_{j}\right)_{j=1}^{\infty}$ such that $\mathcal{K}(x, y)=\sum_{j=1}^{\infty} \sigma_{j} \psi_{j}(x) \psi_{j}(y)$. The function
$\phi(x)=\left(\sqrt{\sigma_{j}} \psi_{j}(x)\right)_{j=1}^{\infty}$ is a map from $\mathcal{X}$ to 1 dimensional square-integrable space $\ell_2$, and defines a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$: For every $t\in\ell_2$, define the corresponding element in $\mathcal{H}$ by $f_t(x) = \langle\phi(x), t\rangle$ with the induced inner product $\left\langle f_{s}, f_{t}\right\rangle_{\mathcal{H}}=\langle t, s\rangle$.

**Assumption 2.**
Without loss of generality, we assume $\mathcal{K}(x,x)$ for all $x\in\mathcal{X}$. We assume that for $0 < p < 1$, the eigenvalues $\sigma_j$ satisfy $G=\sup _{j \geq 1} j^{1 / p} \sigma_{j}$ for some constant $G < 1$, and that the orthonormal eigenfunctions $\psi_{j}(\cdot)$ with $\left\|\overline{\psi}_{j}\right\|_{L_{2}(\mathcal{P})}=1$ are uniformly bounded, i.e., $\sup _{j}\left\|\psi_{j}\right\|_{\infty} \leq A<\infty$. Finally, we assume that the outcomes $Y_i$ are almost surely bounded, $\left|Y_{i}\right| \leq M$.

**Assumption 3.** 
The true CATE function $\tau^{*}(x)=\mathbb{E}\left[Y_{i}(1)-Y_{i}(0) \mid X_{i}=x\right]$ satisfies $\left\|T_{\mathcal{K}}^{\alpha}\left\{\tau^{*}(\cdot)\right\}\right\|_{\mathcal{H}}<\infty$ for some $0<\alpha<1/2$.
(The $\alpha$ power of operator $T_{\mathcal{K}}$ is given by $\alpha$ power of eigenfunctions:  $T_{\mathcal{K}}^{\alpha}\{f(\cdot)\})=\mathbb{E}[\sum_{j=1}^{\infty} \sigma_{j}^\alpha \psi_{j}(\cdot) \psi_{j}(X) f(X)]$.)

The oracle penalized regression estimator is

$$
\tilde{\tau}(\cdot)=\operatorname{argmin}\left(\frac{1}{n} \sum_{i=1}^{n}\left[\left\{Y_{i}-m^{\ast}\left(X_{i}\right)\right\}-\left\{W_{i}-e^{\ast}\left(X_{i}\right)\right\} \tau\left(X_{i}\right)\right]^{2}+\Lambda_{n}\left(\|\tau\|_{\mathcal{H}}\right):\|\tau\|_{\infty} \leq 2 M\right),
$$

and the empirical version estimator obtained by cross-fitting is

$$
\hat{\tau}(\cdot)=\operatorname{argmin}_{\tau \in \mathcal{H}}\left(\frac { 1 } { n } \sum_{i=1}^ { n } \left[\left\{Y_{i}-\hat{m}^{(-q(i))}\left(X_{i}\right)\right\}
-\left\{W_{i}-\hat{e}^{(-q(i))}\left(X_{i}\right)\right\} \tau\left(X_{i}\right)\right]^{2}\\
+\Lambda_{n}\left(\|\tau\|_{\mathcal{H}}\right):\|\tau\|_{\infty} \leq 2 M\right)
$$

The accuracy of any estimator $\tau(\cdot)$ is defined by the regret bound
$$
R(\tau)=L(\tau)-L\left(\tau^{\ast}\right), \quad L(\tau)=\mathbb{E}\left(\left[\left\{Y_{i}-m^{\ast}\left(X_{i}\right)\right\}-\tau\left(X_{i}\right)\left\{W_{i}-e^{\ast}\left(X_{i}\right)\right\}\right]^{2}\right).
$$

Under the regularity assumptions (2,3) and uncounfoundedness (1), assuming $2\alpha<1-p$, and also conditions on the rate of $\hat{m}(x)$ and $\hat{e}(x)$, it is proved that the empirical version estimator obtained via a penalized kernel regression variant of the R-learner, with a properly chosen
penalty of the form $\Lambda_{n}\left(\|\hat{\tau}\|_{\mathcal{H}}\right)$, achieves the same error rate as the oracle version:

$$
R(\hat{\tau})\sim R(\tilde{\tau})=\widetilde{\mathcal{O}}_{P}\left(n^{-\frac{1-2 \alpha}{p+(1-2 \alpha)}}\right):={\mathcal{O}}_{P}\left(n^{-\frac{1-2 \alpha}{p+(1-2 \alpha)}}\log^\beta(n^{-\frac{1-2 \alpha}{p+(1-2 \alpha)}})\right).
$$



<!--References>
------

[https://www.cnblogs.com/gogoSandy/p/11711918.html](https://www.cnblogs.com/gogoSandy/p/11711918.html){:target="_blank"}
[https://zhuanlan.zhihu.com/p/115223013](https://zhuanlan.zhihu.com/p/115223013){:target="_blank"}

<!-- Aren't headings cool?
<!------>
