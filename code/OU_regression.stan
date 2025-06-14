data {
  int<lower=1> N;                         // number of tips with data
  int<lower=1> Nnodes;                    // total number of nodes
  int<lower=1> Nedges;                     // number of edges in the tree
  vector<lower=0, upper=1>[N] x;          // predictor at tips
  array[N] int<lower=0, upper=1> y;       // binary response at tips
  array[Nedges, 2] int<lower=1, upper=Nnodes> edges;     // parent → child
  vector<lower=0>[Nedges] edge_lengths;                 // edge lengths
  int<lower=1, upper=Nnodes> root_node;                 // index of root node
}

parameters {
  vector[Nnodes] z_std;       // standard normal reparam for latent z
  real<lower=0> sigma;        // OU diffusion
  real<lower=0> lambda;       // OU strength
  real mu;                    // OU mean
  real alpha;                 // intercept
  real beta;                  // slope
}

transformed parameters {
  vector[Nnodes] z;

  // root node
  z[root_node] = mu + (sigma / sqrt(2 * lambda)) * z_std[root_node];

  // recursive evolution
  for (e in 1:Nedges) {
    int edge_index = Nedges - e + 1; // reverse order for recursion
    int parent = edges[edge_index, 1];
    int child = edges[edge_index, 2];
    real len = edge_lengths[edge_index];

    real decay = exp(-lambda * len);
    real s = sigma * sqrt(-expm1(-2 * lambda * len) / (2 * lambda));
    real mn = mu + (z[parent] - mu) * decay;

    z[child] = mn + s * z_std[child];
  }
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ lognormal(0, 1);
  lambda ~ lognormal(0, 1);
  mu ~ normal(0, 2);
  z_std ~ normal(0, 1);  // standard normal prior

  // Likelihood
  for (i in 1:N) {
    y[i] ~ bernoulli_logit(alpha + beta * x[i] + z[i]);
  }
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | alpha + beta * x[i] + z[i]);
  }
}
