# Overivew on maximimizing utility for credit models

Let $x$ be a vector of feauters. $y$ is the outcome, means default or not default. 

| y | outcome|
|---|---|
| 1 | No default 
| - 1| Default|

We can use Machine learning to estimate default probability, $f(x)$. To decide whether an individual defaults, we select a threshold, $h$. 
If $f(x)>h$, we reject the loan, and if $\hat{p}(x)<h$, we approve the loan. If no other specified, we usually set $h=0.5$. 
However, doins so is threshold dependent. Differnt $h$ produces different confusion matrix and thus the subsequent measures, accuarcy, F1-Score, etc. Another threshold independent measure is AUC. 

Now, we already have a $f(x)$.

Suppose we are working with Logistic Regression, $f(x)=\sigma(\beta, x) = P(y= - 1 | x) =  \frac{1}{1+exp(-\beta' x)}$. <br> 

$\hat{f} = \arg \max_{\beta} \sum_{i=1}^N ​[y_i​ log( \sigma(\beta,x_i) ​)+(1−y_i​)log(1−\sigma(\beta, x_i)​)].$

#### $u_{a,y}(x)$

|$a$| action|
|---|---|
| +1 | Approve loan|
| -1 | Reject loa| 

$u_{a,y}(x)$ is the utility of action $a$ with outcome $y$. 

$y$: output, is from our data.
$a$: action, depends on $p(x)$, of the form $p(x) > \varespilon=\varespilon(x)$.


Does $u_{a,y}(x)$ relate to 
- \hat{p}(x)?


Tiffany add $u_{a,y}(x)$ 


| $u_{a,y}(x)$ | $y = -1$       | $y = 1$    |
|---|---|---|
| $a = -1$     | $0$            | $0$        |
| $a = 1$      | $-LGD \cdot A$ | $r \cdot A$|


They derive that the optimal decision (i.e., $a$) is determined by whether the true default probability $f(x)$ lies above or below a utility-derived cutoff

$c(x) = \frac{u_{-1,-1}(x) - u_{1,-1}(x)}{u_{1,1}(x) - u_{-1,1}(x) + u_{-1,-1}(x) - u_{1,-1}(x)}=\frac{0 − (−LGD·A)}{r·A − 0 + 0 − (−LGD·A)} = \frac{LGD}{r+LGD},$
where $u_{a,y}(x)$ is the lender's utility from action $a$ when outcome is $y$. The key insight: only the **sign** of $p^*(x) - c(x)$ matters for optimal decisions, not the precise probability — so estimation should target that crossing point, not the conditional probability itself.

#### Decision rule ❓❓❓
- $f(x) > ❓c(x)$, we reject the loan, i.e., $a= -1$ <br>
- $f(x) < ❓c(x)$, we approve the loan, i.e., $a=1$. 




### 4. The role of Maximum Utility (MU) Estimation framework 🤪


Elliott & Lieli (2013) — *Journal of Econometrics*

Generalizes Lieli & White into the semiparametric **Maximum Utility (MU) Estimation** framework. Given i.i.d. data $\{(Y_i, X_i)\}$, the MU estimator solves

$$
\hat{f} \in \arg\max_{f \in \mathcal{F}} \frac{1}{n} \sum_{i} b(X_i)\left[Y_i + 1 - 2c(X_i)\right] \cdot {sign}(f(X_i) - c(X_i)),
$$

---
#### ❓❓❓ The above utiltiy? 
##### Derivation of the Weight Term b(X)[Y + 1 − 2c(X)]

***Setup.*** Define the two correct-decision payoff differentials:

$$G_1 = u_{1,1}(x) - u_{-1,1}(x), \qquad G_2 = u_{-1,-1}(x) - u_{1,-1}(x)$$

Both are strictly positive by Condition 1.

---

##### Step 1: Linearize U in a

Since a ∈ {1, −1}, any function of a can be written uniquely as:

$$U(a,y,x) = A(y,x) + B(y,x)\cdot a$$

where 

$$A(y,x)=\frac{U(1,y,x) + U(-1,y,x)}{2}, B(y,x) = \frac{U(1,y,x) - U(-1,y,x)}{2}$$

So that when $a=1 : A(1,x)+B(1,x)=U(1,y,x)$ ; $a=-1 : A(-1,x)+B(-1,x)=U(-1,y,x)$

Evaluating $B(y,x)$ at each outcome:

$$B(1,x) = \frac{G_1}{2}, \qquad B(-1,x) = -\frac{G_2}{2}$$

---

##### Step 2: Linearize B in y

Since y ∈ {1, −1}, write B(y,x) = p + qy and solve:

$$p + q = \frac{G_1}{2}, \qquad p - q = -\frac{G_2}{2}$$

$$\Rightarrow \quad p = \frac{G_1 - G_2}{4}, \qquad q = \frac{G_1 + G_2}{4}$$

Factoring out (G₁ + G₂)/4:

$$B(y,x) = \frac{G_1+G_2}{4}\left[y + \frac{G_1-G_2}{G_1+G_2}\right]$$

---

##### Step 3: Substitute the definition of c(x)

From equation (2) of Elliott and Lieli (2013):

$$c(x) = \frac{G_2}{G_1+G_2}$$

The remaining fraction simplifies by a pure algebraic identity:

$$\frac{G_1-G_2}{G_1+G_2} = \frac{(G_1+G_2)-2G_2}{G_1+G_2} = 1 - 2c(x)$$

Substituting and noting b(x) ≡ G₁ + G₂:

$$B(y,x) = \frac{b(x)}{4}\big[y + 1 - 2c(x)\big]$$

---

##### Result

Since $U(a,y,x) = A(y,x) + B(y,x)·a$ and A does not depend on a,
maximizing expected utility over $a(·) = sign[g(X)]$ reduces to:

$$\max_{g \in G}\ E_{Y,X}\left\{ b(X)[Y+1-2c(X)]\mathrm{sign}[g(X)]\right\} $$

which is equation (4) of Elliott and Lieli (2013). The factor of 2
arises solely from a scale mismatch: c(x) ∈ (0,1) while y ∈ {−1,1}
has range 2, so multiplying c(x) by 2 aligns the cutoff with the
metric of the outcome variable.

