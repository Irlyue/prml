2.8

(1)
$$
\begin{split}
E_x[x|y] &= \int xp(x|y)dx\\
E_y[E_x[x|y]] &= \int (E_x[x|y])p(y)dy\\
&=\int (\int xp(x|y)dx)p(y)dy \\
&=\int(\int xp(x|y)p(y)dx)dy  \text{ #push in }p(y)\\
& = \int\int xp(x,y)dxdy\text{ #use }p(x,y)=p(x|y)p(y)\\
&=E[x]
\end{split}
$$
2.26

证明$(A+BCD)^{-1}=A^{-1}-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}DA^{-1}$

首先，两边右乘$(A+BCD)$，于是有：
$$
\begin{split}
I& =A^{-1}(A+BCD)-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}DA^{-1}(A+BCD)\\
&= I+A^{-1}BCD-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}D-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}DA^{-1}BCD\\
\end{split}
$$
也就是说这一大串$A^{-1}BCD-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}D-A^{-1}B(C^{-1}+DA^{-1}B)^{-1}DA^{-1}BCD=0$

把相同因子提取出来有：
$$
A^{-1}B-->(C-(C^{-1}+DA^{-1}B)^{-1}-(C^{-1}+DA^{-1}B)^{-1}DA^{-1}BC)<--D
$$
即有$C-(C^{-1}+DA^{-1}B)^{-1}-(C^{-1}+DA^{-1}B)^{-1}DA^{-1}BC=0$

两边同时左乘于$(C^{-1}+DA^{-1}B)$，容易证明上面的式子成立，得证。

