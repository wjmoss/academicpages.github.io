---
title: 'Learning the difference DAG between structural equation models'
date: 2022-02-08
permalink: /posts/2022/02/diffDAG/
tags:
  - structural equation model
  - directed acyclic graph
  - theory
---

The paper "Direct Learning with Guarantees of the Difference
DAG Between Structural Equation Models" by Asish Ghoshal, Kevin Bello and Jean Honorio studies the problem of directly estimating the structural difference between two structural equation models (SEMs) with the same topological ordering. 

1.Introduction
======
The paper has two main contributions: 1. It proposes an algorithm that recover the difference DAG with $\epsilon$-accurate in $O(d^2\log(p))$ samples. If all the error variances do not change between the two SEMs, the accuracy means true difference DAG. Otherwise the assuracy is for orientable edges and the skeleton. 2. It gives the fundamental limits of sample size for $\epsilon$ error bound: $\Omega(d'\log(p/d'))$. Where $d$ is the maximal number of edges in the difference of the moralized sub-graphs of the two SEMs, and $d'$ is the maximal number of parent set size for one node in the difference DAG.

2.Setups
======
Let $X=(X_1, ... , X_p)$ be a $p$-dimensional vector and $\epsilon=(\epsilon_1, ... ,\epsilon_p)$ be independent random errors. An SEM is given by the form

$$
X = B X + \epsilon,
$$

where $B$ is the autoregression matrix (transpose of adjacency matrix, $B$ encodes a DAG and $B_{ij}$ corresponds to the edge $i \rightarrow j$) and $\epsilon$ has diagonal covariance matrix  $D=\text{diag}(\{\sigma_i^2\})$. The random vector $X$ has covariance matrix $\Omega=(I-B)^{-1}D(I-B)^{-T}$. We denote the SEM by the tuple $(B,D)$.

The goal is to recover the structural difference between the two DAGs correponding to SEMs $(B^{(1)},D^{(1)})$ and $(B^{(2)},D^{(2)})$: $\text{supp}(\Delta_B):=\text{supp}(B^{(1)}-B^{(2)})$, given the data samples $X^{(1)}\in\mathbb{R}^{n_1\times p}$, $X^{(2)}\in\mathbb{R}^{n_2\times p}$. For that we need 2 important assumptions:

1. The individual DAG might be dense, but the diffenrence is sparse ($\ll p$ non-zero entries in each row / column of $\Delta_B$).
2. The (unknown) topological ordering of the two SEMs remains consistent.

3.Results
======
3.1 Preliminaries
------
The novelty of their algorithm relies on the equal variance assumption between two SEMs (the mechanism of errors generation remains the same, naturally, and as a special case). In this case the algorithm use this proposition to identify terminal vertice iteratively:

**Proposition 1**
Fixing a node $i$, if we have $$B^{(1)}_{ji}=B^{(2)}_{ji}, D^{(1)}_{ii}=D^{(2)}_{ii}, D^{(1)}_{jj}=D^{(2)}_{jj}$$ for any node $j$, then $$(\Delta_\Omega)_{ii}=0$$ ($\Delta_\Omega$ is the difference of two precision matrices). Furthermore, $i$ is a terminal vertex in the difference DAG $G = ([p], \Delta)$ with $$\Delta = supp(B^{(1)}-B^{(2)})$$.

This proposition gives a sufficient condition for $(\Delta_\Omega)_{ii}=0$, which is again a sufficient condition for terminal vertex. (ps: $$\Omega_{ii}=\sigma_{i}^2+\sum_{i \rightarrow k}{B_{ki}}^2\sigma_{k}^2$$, in the polynomial equation sense, the second sufficient is also necessary since the equation for superscripts $1,2$ requires all parameters equaling.)

A well-known result for SEM marginalization is that, the new SEM obtained by removing a terminal vertex $i$ from the original one $(B,D)$ is given by $(B_{-i,-i},D_{-i,-i})$, which means the submatrix except the$i$'th row and $i$'th column. This is all what we need from marginalization theory, in order to perform the algorithm. However, the paper also gives the full result of arbitrary marginalzation, as a lemma.

**Lemma 1**
The SEM obtained by removing a subset of vertices $U\subset[p]$, i.e., the SEM over $X_{[p]\backslash U}$, is given by $$(\tilde{B}; \tilde{D})$$ with $$\tilde{D}=\text{diag}(\{\tilde{\sigma}_{j}\}_{j\in[p]\backslash U})$$ and

$$
\tilde{\sigma}_{j}^{2}=\sigma_{j}^{4}\left\{\sigma_{j}^{2}-B_{j, U_{j}}\left(\Omega_{U_{j}, U_{j}}^{\mathcal{A}_{j}}\right)^{-1}\left(B_{j, U_{j}}\right)^{\top}\right\}^{-1}, \quad \tilde{B}_{j, k}=\frac{\tilde{\sigma}_{j}^{2}}{\sigma_{j}^{2}}\left\{B_{j, k}-B_{j, U_{j}}\left(\Omega_{U_{j}, U_{j}}^{\mathcal{A}_{j}}\right)^{-1}\left(\Omega_{U_{j}, k}^{\mathcal{A}_{j}}\right)\right\}
$$

$$\forall j\in[p]\backslash U$$ and $$k\in \mathcal{A}_j$$, where $$\mathcal{A}_j$$ denotes the ancestors of node $j$. Also, $$U_{j}=\mathcal{A}_{j} \cap U$$, and $$\Omega^{\mathcal{A}_{j}}$$ is the  precision matrix for $$X_{\mathcal{A}_{j}}$$ . Finally, for $$k \notin \mathcal{A}_{j}, \tilde{B}_{j, k}=0$$.


3.2 Algorithm
------
The high level sketch of the algorithm contains four main steps.
1. Remove the invariant vertices, that is, vertices for which the corresponding rows and columns in the difference of precision matrix is all zeros.
2. Estimate the (partial) topological ordering over the remaining vertices in the difference DAG
3. orient the edges present in the difference of precision matrix according to the ordering
4. Remove the "extra" edges according to faithfulness

The population version algorithm requires this assumption

**Assumption 1**.
Let $(B^{(1)};D^{(1)})$ and $(B^{(2)};D^{(2)})$ be two SEMs with the difference DAG given by $G = ([p], \Delta)$, where $\Delta = supp(B^{(1)} -B^{(2)})$, and difference of precision matrix given by $\Delta_\Omega$. Let $$U=\left\{i \in[p] \mid\left(\Delta_{\Omega}\right)_{i, *}=0\right\}$$ and let $$V=[p]\backslash U$$,.Then the two SEMs satisfy the following assumptions:

(i) For $i \in U$, the edges and noise variances are invariant.

(ii) For each $(i, j)\in \Delta$, and $\forall S \subset [p], i,j\in S$, we have that $$\operatorname{corr}^{(1)}\left(X_{i}, X_{j} \mid X_{S'}\right) \neq \operatorname{corr}^{(2)}\left(X_{i}, X_{j} \mid X_{S'}\right)$$, where $$S'=S \backslash\{i, j\}$$.

The condition (i) says that, if all undirected edges incident on a vertex in the moral graph remain the same, then the directed edges incident
on the node remains invariant. The condition (ii) is teh faithfulness assumption in difference DAG, if any edge $(i,j)$ changes, it will at least make difference on the precision matrix (or partial correlation for some submodel).  Indeed, we have $$\operatorname{cov}(X_i,X_j\mid X_{S'})=(\Sigma_{ij,ij}-\Sigma^S_{ij,S'}(\Sigma_{S',S'})^{-1}\Sigma_{S',ij})_{ij}=[(\Omega^S)^{-1}]_{ij}$$, and 

$$
\operatorname{corr}(X_i,X_j\mid X_{S'})=-\frac{[(\Omega^S)^{-1}]_{ij}}{[(\Omega^S)^{-1}]_{ii}[(\Omega^S)^{-1}]_{jj}}.
$$

It is really confusing that the paper does not state the error propagation between precision matrix (or covariance matrix) and partial correlations. 

The picture of detailed algorithm is screenshotted from the original paper.

![diffDAG estimation algorithm](/images/diffDAG-alg.PNG "diffDAG estimation algorithm")

In the algorithm, the function ComputeOrder finds terminal vertices iteratively from the bottom layer. The OreintEdges function orient the edges (non-zero $\Delta_\Omega$_{ij}) from higher layers to lower layers, and finally add edges with no orientation in the rest part (unknown topological ordering). The Prune function applies Assumption 1.(ii) to check which edges to delete.

3.3 Finite-sample guarantees
------
In application scenarios, the sampling error cannot be avoid and the algorithm needs to be slightly modified. First, the difference of precision matrix can be computed by $\Sigma^{(1)}\Delta_\Omega\Sigma^{(2)}=\Sigma^{(2)}-\Sigma^{(1)}$, which also works in finite sample case. Another paper [] proposed an estimator for the difference of precision matrix by solving this optimization problem:

$$
\widehat{\Delta}_{\Omega}=\underset{\Delta_{\Omega}}{\operatorname{argmin}}\left\|\Delta_{\Omega}\right\|_{1} \text { subject to }\left|\widehat{\Sigma}^{(1)}\left(\Delta_{\Omega}\right) \widehat{\Sigma}^{(1)}-\widehat{\Sigma}^{(2)}+\widehat{\Sigma}^{(1)}\right|_{\max } \leq \lambda_{n}.
$$

Denoting $\beta=\operatorname{vec}(\Delta_\Omega)$, it can be rewritten as

$$
\widehat{\beta}=\underset{\beta}{\operatorname{argmin}}\|\beta\|_{1} \text { subject to }\left|\left(\widehat{\Sigma}^{(2)} \otimes \widehat{\Sigma}^{(1)}\right) \beta-\operatorname{vec}\left(\widehat{\Sigma}^{(1)}-\widehat{\Sigma}^{(2)}\right)\right|_{\max } \leq \lambda_{n} .
$$

(ps: $\operatorname{vec}(AXB^T)=(B\otimes A)\operatorname{vec}(X)$)

For finite sample version algorithm analysis, we require the oracle assumption of difference of precision matrix estimation, which is from Lasso literatures.

**Assumption 2**
Given $X^{(1)}\in\mathbb{R}^{n_1\times p}$, $X^{(2)}\in\mathbb{R}^{n_2\times p}$, there exists an estimator $$\widehat{\Delta}_\Omega$$, s.t. $$P\left\{\left\vert\widehat{\Delta}_{\Omega}-\Delta_{\Omega}\right\vert_{\max } \leq \varepsilon\right\} \geq 1-\delta$$, if $$n_{1} \geq \eta_{1}(\varepsilon, \delta)$$ and $$n_{2} \geq \eta_{2}(\varepsilon, \delta)$$ for some $$\varepsilon,\delta>0$$ and function $$\eta_1,\eta_2$$.

And also the finite sample version of Assumption 1.

**Assumption 3**
Let $(B^{(1)};D^{(1)})$ and $(B^{(2)};D^{(2)})$ be two SEMs with the difference DAG given by $G = ([p], \Delta)$, where $\Delta = supp(B^{(1)} -B^{(2)})$, and difference of precision matrix given by $\Delta_\Omega$. Let $$U=\left\{i \in[p] \mid\left(\Delta_{\Omega}\right)_{i, *}=0\right\}$$ and let $$V=[p]\backslash U$$,.Then the two SEMs satisfy the following assumptions:

(i) For $i \in U$, the edges and noise variances are invariant.

(ii) For each $(i, j)\in \Delta$, and $$\forall S \subset [p], i,j\in S$$, we have that $$\vert\operatorname{corr}^{(1)}\left(X_{i}, X_{j} \mid X_{S^{\prime}}\right) - \operatorname{corr}^{(2)}\left(X_{i}, X_{j} \mid X_{S^{\prime}}\right)\vert\geq 2\varepsilon$$, for $S'=S \backslash\{i, j\}$ and for some $\varepsilon>0$.

Then we have  the soundeness theorem under oracle assumption. 

**Theorem 2**
Under Assumption 2,3. Let $$\Delta_G=([p],\Delta^*)$$ be the true difference DAG with $$\Delta^*=\operatorname{supp}(B^{(1)}-B^{(2)})$$. Given $$\widehat{\Sigma}^{(1)}$$, $$\widehat{\Sigma}^{(2)}$$, $n_1,n_2$, and $\varepsilon>0$ as input, the finite sample learning algorithm returns $\Delta$ such that $$\text{skel}(\Delta)=\text{skel}(\Delta^*)$$ with probability as least $1-\delta$ if $$n_{1} \geq \eta_{1}(\varepsilon, \delta)$$ and $$n_{2} \geq \eta_{2}(\varepsilon, \delta)$$. Furthermore, if $D^{(1)}=D^{(2)}$ then $$\Delta=\Delta^*$$.

The proof seems to omit the derivation of error propagation and mentions that $$\vert\widehat{\Delta}^S_{\Omega}-{\Delta}^S_{\Omega}\vert\leq\varepsilon$$ with probability $\geq 1-\delta$ simutaneously over all $S\subset [p]$, which are weird (although the sample size condition in Assumption 2 does not contain $p$...).

Finally, combining the Lasso results, here is the core corollary for sample size guarantee condition.

**Theorem 3** (Auxilirary, adapted from [])
Define $$K_{\max }^{\circ} \stackrel{\text { def }}{=} \max _{(i, j) \neq(k, l)}\left|\Sigma_{i, j}^{(1)} \Sigma_{k, l}^{(2)}\right|$$ and $$K_{\min }^{\mathrm{d}} \stackrel{\text { def }}{=} \min _{i} \Sigma_{i, i}^{(1)} \Sigma_{i, i}^{(2)}$$. Let $$\lambda_{\min }(\cdot)$$ denote the
minimum eigenvalue of a matrix. If $$K_{\max }^{\circ} \leq \frac{\lambda_{\min }\left(\Sigma^{(1)}\right) \lambda_{\min }\left(\Sigma^{(2)}\right)}{2\left\|\Delta_{\Omega}\right\|_{0}}$$, the regularization parameter, $\lambda_n$, and the number of samples, $n$, satisfy the following conditions:

$$
n \geq \frac{C^{2}}{\left(K_{\mathrm{min}}^{\mathrm{d}} \varepsilon\right)^{2}} \log \frac{2 p}{\delta} \quad \text { and } \quad \lambda_{n} \geq C \sqrt{\frac{1}{n} \log \frac{2 p}{\delta}}
$$

where $C$ is a constant that depends linearly on $$\vert\Delta_\Omega\vert_1$$, $$\vert\Sigma^{\kappa}\vert_{\max}$$ and $$\max _{\kappa, i} \sum_{i, i}^{(\kappa)}$$, then with probability at least $1-\delta$ we have that $$\vert\Delta_{\Omega}-\widehat{\Delta}_{\Omega}\vert_{\max } \leq \varepsilon$$.

**Corollary**
Let $$d=\max_{S\in[p]}\|\Delta^{S}_\Omega\|_0$$ (max diff in moral subgraphs). If $$\min(n_1,n_2)=O\left(\left(\frac{d^{2}}{\varepsilon^{2}}\right) \log \left(\frac{p}{\delta}\right)\right)$$, $$K_{\max }^{\circ} \leq \frac{\lambda_{\min }\left(\Sigma^{(1)}\right) \lambda_{\min }\left(\Sigma^{(2)}\right)}{2 d}$$, and $$\lambda_{n}=\Omega\left(\sqrt{\frac{1}{n} \log \frac{2 p}{\delta}}\right)$$, where the constant $$K_{\max }^{\circ}$$ is defined in Theorem 3, and the true difference DAG satisfies Assumption 3, then $\Delta$ (or $$\text{skel}(\Delta)$$) is correctly identified with probability as least $1-\delta$.

(just plugging in the bound of $$K_{\max }^{\circ}$$...)


4.Fundamental limits
======
TBD





<!--References>
------

[https://www.cnblogs.com/gogoSandy/p/11711918.html](https://www.cnblogs.com/gogoSandy/p/11711918.html){:target="_blank"}
[https://zhuanlan.zhihu.com/p/115223013](https://zhuanlan.zhihu.com/p/115223013){:target="_blank"}

<!-- Aren't headings cool?
<!------>
