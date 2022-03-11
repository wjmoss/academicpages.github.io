---
title: 'Some notes about KL divergence'
date: 2022-01-05
permalink: /posts/2022/03/kldiv/
tags:
  - 

---


1.$f$-divergence
======

KL divergence is one special case of f-divergences. Before discussing about the KL divergence in detail, we first summarize some results in more general setup.

 The f-divergences is a class of measurements that measure the dissimilarity between two distributions.  

Let $f:\mathbb{R}^\ast\mapsto\mathbb{R}$ be a convex function (i.e. Jensen's inequality $$\mathbb{E}[f(x)] \geq f(\mathbb{E}[x])$$) with $f(0)=1$, for any probability density $p,q$ such that $\text{supp}(p)\subseteq\text{supp}(q)$, the f-divergence between $p$ and $q$ is defined by 

$$D_{f}(P \| Q)=\int q(x) f\left(\frac{p(x)}{q(x)}\right) d x.$$

Different choices of $f$ give different expression of the distance. Some common choices and corresponding distance formulas are shown in the table.

$$
\begin{array}{c|c|c}\hline 
\textbf{Distance} & \textbf{formula} & \textbf{corresponding }f\\ 
\hline 
\text{Total variation} & \frac{1}{2}\int | p(x) - q(x)| dx & \frac{1}{2}|u  - 1|\\ 
\hline 
\text{Kullback-Leibler (KL) divergence} & \int p(x)\log \frac{p(x)}{q(x)} dx & u \log u\\ 
\hline 
\text{Reverse KL divergence} & \int q(x)\log \frac{q(x)}{p(x)} dx & - \log u\\ 
\hline 
\text{Pearson }\chi^2\text{-divergence} & \int \frac{(q(x) - p(x))^{2}}{p(x)} dx & \frac{(1  - u)^{2}}{u}\\ 
\hline 
\text{Neyman }\chi^2\text{-divergence} & \int \frac{(p(x) - q(x))^{2}}{q(x)} dx & (u  - 1)^{2}\\ 
\hline 
\text{Hellinger distance} & \int \left(\sqrt{p(x)} - \sqrt{q(x)}\right)^{2} dx & (\sqrt{u} - 1)^{2}\\ 
\hline 
\text{Jeffrey distance} & \int (p(x) - q(x))\log \left(\frac{p(x)}{q(x)}\right) dx & (u - 1)\log u\\ 
\hline 
\text{Jensen-Shannon divergence} & \frac{1}{2}\int p(x)\log \frac{2 p(x)}{p(x) + q(x)} + q(x)\log \frac{2 q(x)}{p(x) + q(x)} dx & -\frac{u  + 1}{2}\log \frac{1  + u}{2} + \frac{u}{2} \log u\\ 
\hline 
\end{array}
$$

The domain $\mathbb{R}^\ast$ and the support condition ensure that $f(p(x)/q(x))$ is well-defined. The condition $f(0)=1$ implies that $D_f(P\|P)=0$. The convexity of $f$ derives non-negativeness of $D_f$.

$$
\begin{aligned}
\int q(x) f\left(\frac{p(x)}{q(x)}\right) d x &=\mathbb{E}_{x \sim q(x)}\left[f\left(\frac{p(x)}{q(x)}\right)\right] \\
& \geq f\left(\mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)}\right]\right) \\
&=f\left(\int q(x) \frac{p(x)}{q(x)} d x\right) \\
&=f\left(\int p(x) d x\right) \\
&=f(1)=0.
\end{aligned}
$$



2.KL divergence and Maximum likelihood
======
Suppose that we have a bunch of data $(X_1,\dots,X_n)$ from the unknown distribution $p_{data}$, and we want to fit the data by a distribution $p_\theta$ in a parametric family. It is natural to consider minimizing some $f$-divergence between $p_\theta$ and the empirical distribution. We can regard the integration as the expectation of some $h(X)$ with $X\sim p_{data}$, and the empirical average of $h(X_1),\dots,h(X_n)$ is an unbiased "estimator" (we use the quotation marks here because the value may not be able to computed from data) of $D_{f}(p_{data} \| p_\theta)$. 

$$
\begin{aligned}
D_{f}(p_{data} \| p_\theta)&=\int p_\theta(x) f\left(\frac{p_{data}(x)}{p_\theta(x)}\right)dx\\
&=\int p_{data}(x)\frac{p_\theta(x)}{p_{data}(x)} f\left(\frac{p_{data}(x)}{p_\theta(x)}\right)dx\\
&=\mathbb{E}_{p_{data}}\left[\frac{p_\theta(X)}{p_{data}(X)} f\left(\frac{p_{data}(X)}{p_\theta(X)}\right)\right]\\
&\approx \frac{1}{n}\sum_{i=1}^n\frac{p_\theta(X_i)}{p_{data}(X_i)} f\left(\frac{p_{data}(X_i)}{p_\theta(X_i)}\right).
\end{aligned}
$$

The density $p_\theta$ can be evaluated, but we don't know the true distribution and hence $p_{data}$ is unknown. Of course we don't want to get involved in density estimation problem of $p_{data}$. (if we can do that, why not just directlt estimating the density?) It follows that the $p_{data}$ term should ideally be cancelled by a suitable choice of the function $f$. After examining all functions in the table above, the only one works for the average estimator is $f(u)=u\log(u)$, which corresponds to KL divergence.

Then we have the empirical version of KL divergence

$$\widehat{D}_{KL}(p_{data} \| p_\theta)={D}_{KL}(p_{emp} \| p_\theta)= \frac{1}{n}\sum_{i=1}^n \left(\log(p_{data}(X_i))-\log(p_\theta(X_i))\right).$$

Since each $\log(p_{data}(X_i)$ is a constant w.r.t. $\theta$, the problem of minimizing $\widehat{D}_{KL}(p_{data} \| p_\theta)$ is equivalent to maximizing $\frac{1}{n}\sum_{i=1}^n\log(p_\theta(X_i))$ --- the obtained estimator $\hat{\theta}$ is just the maximum likelihood estimator. Among all $\theta$ in the parameter space, the value $\theta=\hat{\theta}$ achieves the maximum (log-)likelihood value w.r.t. density $p_\theta$ given the observed data. In the sense of $f$-divergence, the KL divergence of $p_{emp}$ from $p_\theta$ ($D_{KL}(p_{emp}\|p_\theta)$) is minimized at $\theta=\hat{\theta}$.



3.Some properties
======
Now let's look at the formula of KL divergence again.

$$D_{KL}(p\|q)=\int p(x)\log \frac{p(x)}{q(x)} dx=\int p(x)\log p(x)dx-\int p(x)\log q(x)dx.$$

$$D_{KL}(q\|p)=\int q(x)\log \frac{q(x)}{p(x)}dx$$

For a fixed $p$, if we want the distance $D_{KL}(p\|q)$ to be small, then $q(x)$ must be small in the area where $p(x)$ is large; and the value of $q(x)$ is not so important in the area where $p(x)$ is small. Conversely, if we want $D_{KL}(q\|p)$ to be small, then $q(x)$ must be small in the area where $p(x)$ is small, and the value of $q(x)$ is not so important in the area where $p(x)$ is large. One can also refer to a picture from "Machine Learning: A Probabilistic Perspective".

For two multivariate normal distribution $P_1=N(\mu_1,\Sigma_1)$ and $P_2=N(\mu_2,\Sigma_2)$, the KL divergence of $P_1$ from $P_2$ is

$$D_{KL}(P_1\|P_2)=\frac{1}{2}\left[\text{Tr}(\Sigma_2^{-1}\Sigma_1)-\log(\Sigma_2^{-1}\Sigma_1)+(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2)-n\right].$$

KL divergence is not a distance, since it is not symmetric ($D_{KL}(p\|q)\neq D_{KL}(q\|p)$). It does not satisfy triangular inequality either. Use the formula for distance of normal distributions, we can show that

$$D_{KL}(N(0,4)\|N(0,1))>D_{KL}(N(0,4)\|N(0,2))+D_{KL}(N(0,2)\|N(0,1)).$$


4.In practice
======
KL divergence, as the form of log-likelihood, is widely used in Statistics. Undergraduate Statistics textbooks seem rarely(? just my feelings...) mention KL divergence, but they all talk about likelihood inference. The $\log$ term becomes a rational function after computing derivative, and the first order condition equations consist of a polynomial system after reduction of fractions. To study the properties of those polynomials (e.g. the number of solutions in $\mathbb{C}^d$, called maximum likelihood degree), is the task in the area of Algebraic Statistics.

In the area of Machine Learning, I cannot say much about it. KL divergence is criticized because it is unbounded, not symmetric, and the $\log$ term may cause exploding gradient in back propagation. But the evaulation of KL divergence is easy (comparing to Wasserstein distance etc...) and the usage of maximum likelihood estimation is natural. At least I know that the ELBO in variational inference also adopt KL divergence (heh I will not discuss about it here (-: ).



<!-- Aren't headings cool?
<!------>
