# Install pylearn-mulm

[pylearn-mulm](https://github.com/neurospin/pylearn-mulm)

```
wget https://github.com/neurospin/pylearn-mulm/archive/refs/heads/master.zip
unzip master.zip
ln -s pylearn-mulm-master/mulm ./
rm master.zip
```

https://github.com:neurospin/pylearn-mulm.git 

# Script to build Comparisons tables for publications is here:

/home/ed203246/git/2025_educhesnay_neuroimaging-ml-sklearn/2024_petiton_biobd-bsnip-predict-dx

# Install iterative-stratification

```
wget https://github.com/trent-b/iterative-stratification/archive/refs/heads/master.zip
unzip master.zip
ln -sf iterative-stratification-master/iterstrat ./
rm master.zip
```

# Scaling vs tiv regression (g = "global" tiv)

## scaling 
$x_s = x / g$
Hypothesis: $x = g * s$, (s is the relevant signal to be retrieved, to match x_s) global volume has a global multiplicative effect 

## TIV regression

$x_r = x - a g$ where a = coef of lr of x on g, 

Hypothesis $x = a g + s$ (s is the relevant signal to be retrieved, to match x_r), global volume has a global additive effect 

# Statistics

## y_adj

y = b0 + b1 x1 + b2 x2 + b3 x3 + e

visu of partial residuals x1 when adjusted for all other= 
y_adj = y - b0 - b2 x2 - b3 x3

y_adj = (b0 + b1 x1 + b2 x2 + b3 x3 + e) - b0 - b2 x2 - b3 x3
y_adj = b1 x1 + e


