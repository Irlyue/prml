[TOC]
# 常用的概率分布

## Bernoulli Distribution

举个抛硬币的例子，假设硬币正反两面向上的概率均为0.5,抛硬币的结果为一个随机变量，而且这个随机变量正好可以用**伯努利分布**来刻画。
$$
\Large{
    \begin{split}
    Bern(x|\mu) &= \mu^x(1-\mu)^{1-x}\\
    E[x] &= \mu\\
    Var[x]&=\mu(1-\mu)
    \end{split}
}
$$

## Binomial Distribution

**二项分布**可以认为是伯努利分布的一个延展。还是以抛硬币为例，这次我们不单单抛一个硬币，而是同时抛多个硬币，然后我们统计硬币朝上的个数，这种情形刚好可以用二项分布来刻画。
$$
\Large{
    \begin{split}
    x\sim Bin(m|\mu, n) &= 
    \begin{pmatrix}
    n\\
    m
    \end{pmatrix}
   \mu^m(1-\mu)^{n-m}\\
   E[x] &= n\mu \\
   Var[x]&=n\mu(1-\mu)
    \end{split}
}
$$
  