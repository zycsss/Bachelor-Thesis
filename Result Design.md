I would use **three main plots**: one for accuracy, one for speed, and one for scaling. Then optionally one detail plot showing per-sensor allocation.

## 1. Accuracy plot: CVX vs NN `T_max`

Use this to show the NN result is close to the CVX optimum.

**Plot type:** scatter plot
**x-axis:** `T_max_CVX`
**y-axis:** `T_max_NN`

Add the diagonal line:

```text
y = x
```

Interpretation:

```text
Points close to the diagonal mean NN ≈ CVX.
Points above the diagonal mean NN is worse than CVX.
```

This is probably the most important plot.

Recommended labels:

```text
x-axis: CVX max completion time [s]
y-axis: NN max completion time [s]
title: NN vs CVX synchronization time
```

Also report:

```python
relative_gap = (T_max_NN - T_max_CVX) / T_max_CVX
```

------

## 2. Gap distribution plot

Use this to show the error is usually small.

**Plot type:** histogram or box plot
**x-axis:** relative gap [%]
**y-axis:** number of test samples

where:

```python
gap_percent = 100 * (T_max_NN - T_max_CVX) / T_max_CVX
```

Recommended labels:

```text
x-axis: Relative gap to CVX optimum [%]
y-axis: Number of samples
title: Distribution of NN optimality gap
```

This is better than only reporting the average, because it shows whether the NN is consistently close or only good on average.

Also report:

```python
mean_gap
median_gap
95th_percentile_gap
max_gap
```

------

## 3. Runtime comparison plot

Use this to show the NN is faster.

**Plot type:** bar chart
**x-axis:** method
**y-axis:** average inference/solve time per sample [ms]

Methods:

```text
CVX/MOSEK
NN only
NN + analytical f
```

If your final model uses analytical `f`, then use:

```text
CVX/MOSEK
NN + analytical f
```

Recommended labels:

```text
x-axis: Method
y-axis: Runtime per sample [ms]
title: Runtime comparison
```

Also compute speedup:

```python
speedup = runtime_cvx / runtime_nn
```

Report it as:

```text
NN is X× faster than CVX.
```

This is the plot that directly supports your speed claim.

------

## 4. Scaling plot with number of sensors `K`

Use this to show complexity/scalability.

**Plot type:** line plot
**x-axis:** number of sensors `K`
**y-axis:** runtime per sample [ms]

Lines:

```text
CVX/MOSEK
NN + analytical f
```

Recommended labels:

```text
x-axis: Number of sensors K
y-axis: Runtime per sample [ms]
title: Runtime scaling with sensor count
```

This is useful because your method should scale roughly linearly with `K`, while CVX should grow much faster.

Suggested `K` values:

```python
K_values = [4, 8, 16, 32, 64]
```

If CVX becomes too slow, use:

```python
K_values = [4, 8, 12, 16]
```

------

## 5. Optional detail plot: per-sensor stacked completion time

Use this to show the solution structure for one representative scenario.

**Plot type:** stacked bar chart
**x-axis:** sensor index
**y-axis:** completion time [s]

Stack components:

```text
T_comp
T_tr
T_dt
```

Make two versions side by side or two separate plots:

```text
CVX
NN + analytical f
```

This shows whether both methods balance the sensors similarly.

Recommended labels:

```text
x-axis: Sensor index
y-axis: Completion time [s]
title: Per-sensor completion time decomposition
```

This is similar to the figure you already plot.

------

## 6. Optional allocation comparison plot

Use this if you want to show that the NN learned similar resource allocation.

**Plot type:** grouped bar chart
**x-axis:** sensor index
**y-axis:** allocated resource

Make one for bandwidth:

```text
b_CVX vs b_NN
```

and one for compute:

```text
f_CVX vs f_NN
```

Recommended labels:

```text
x-axis: Sensor index
y-axis: Bandwidth allocation [Hz]
title: Bandwidth allocation comparison
x-axis: Sensor index
y-axis: DT compute allocation [Hz]
title: DT compute allocation comparison
```

But do not rely on this as the main evidence. The final objective `T_max` matters more than matching `b` and `f` exactly.

------

## Best set for your thesis/report

I recommend these four:

| Plot               | x-axis                | y-axis                    | Purpose                  |
| ------------------ | --------------------- | ------------------------- | ------------------------ |
| CVX vs NN accuracy | `T_max_CVX [s]`       | `T_max_NN [s]`            | Show near-optimal result |
| Gap distribution   | relative gap `[%]`    | sample count              | Show consistency         |
| Runtime comparison | method                | runtime per sample `[ms]` | Show speedup             |
| Runtime scaling    | number of sensors `K` | runtime per sample `[ms]` | Show scalability         |

If you only have space for two plots, use:

```text
1. CVX vs NN T_max scatter
2. Runtime comparison bar chart
```

Those directly prove the main claim:

```text
NN + analytical f is much faster while producing almost the same T_max as CVX.
```