
# Convex Optimization


## Count the optimization variables

Now count the unknowns the leader is optimizing:

- $b_1,\dots,b_K$: $K$ bandwidth variables
- $f_1^{DT},\dots,f_K^{DT}$: $K$ compute-allocation variables
- $t$: 1 epigraph variable

So the total number of decision variables is
$$
n = 2K + 1.
$$
This variable count is what drives the linear-algebra cost inside a generic interior-point solver.

## Count the constraints

After the epigraph reformulation, the main constraints are:

- $K$ constraints of the form
  $$
  T_k^{total}(b_k,f_k^{DT};\lambda_k)\le t
  $$

- 1 total-bandwidth constraint

- 1 total-compute constraint

So there are roughly
$$
m = K + 2
$$
main convex constraints.

Depending on the implementation, you may also include domain constraints like $b_k\ge 0$, $f_k^{DT}>0$, but these do not change the asymptotic conclusion.

## Infer the cost of one interior-point step

For a generic dense convex solver, the dominant cost per Newton step is solving a linear system whose dimension scales with the number of primal/dual variables. In standard complexity estimates, this gives a cubic dependence on the problem size:
$$
O(n^3).
$$
Since here $n = 2K+1$,
$$
O((2K+1)^3) = O(K^3).
$$
This is the usual estimate for **one solver iteration** of the leader problem.

> **one interior-point/Newton iteration inside CVX**, the complexity is:
> $$
> \boxed{O(K^3)}.
> $$

## Infer the cost of solving the whole leader problem once

Interior-point methods usually require on the order of
$$
O(\sqrt{m}\log(1/\varepsilon))
$$
Newton steps to reach accuracy $\varepsilon$, where $m$ is the number of constraints.

Here $m \approx K+2$, so the number of solver iterations is approximately
$$
O(\sqrt{K}\log(1/\varepsilon)).
$$

Multiplying by the cost per Newton step:

$$
O(K^3)\cdot O(\sqrt{K}\log(1/\varepsilon))=
O(K^{3.5}\log(1/\varepsilon)).
$$

**one outer iteration of Algorithm 1**, where the leader subproblem is fully solved once, then the leader-side cost is approximately
$$
\boxed{O(K^{3.5}\log(1/\varepsilon))}.
$$