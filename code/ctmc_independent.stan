functions {
  real safe_log(real x) {
    return log(fmax(x, 1e-12));  // avoid log(0) by clipping small values
  }

  row_vector safe_log_row(row_vector x) {
    row_vector[rows(x)] y;
    for (i in 1:rows(x)) {
      y[i] = safe_log(x[i]);
    }
    return y;
  }
}


data {
  int<lower=1> N;
  int<lower=1> Nnodes;
  int<lower=1> Nedges;
  array[N] int<lower=0, upper=1> x;
  array[N] int<lower=0, upper=1> y;
  array[Nedges, 2] int<lower=1, upper=Nnodes> edges;
  vector<lower=0>[Nedges] edge_lengths;
  int<lower=1, upper=Nnodes> root_node;
}

parameters {
  vector<lower=0>[2] rates;
  simplex[2] pi_1;
  simplex[2] pi_2;
}

transformed parameters {
  // Q matrices
  matrix[2, 2] Q1;
  matrix[2, 2] Q2;

  Q1[1,1] = -rates[1] * pi_1[2];
  Q1[1,2] =  rates[1] * pi_1[2];
  Q1[2,1] =  rates[1] * pi_1[1];
  Q1[2,2] = -rates[1] * pi_1[1];

  Q2[1,1] = -rates[2] * pi_2[2];
  Q2[1,2] =  rates[2] * pi_2[2];
  Q2[2,1] =  rates[2] * pi_2[1];
  Q2[2,2] = -rates[2] * pi_2[1];

  matrix[Nnodes, 2] loglikelihood_x;
  matrix[Nnodes, 2] loglikelihood_y;

  for (n in 1:Nnodes) {
    loglikelihood_x[n] = rep_row_vector(negative_infinity(), 2);
    loglikelihood_y[n] = rep_row_vector(negative_infinity(), 2);
  }

  for (i in 1:N) {
    loglikelihood_x[i, x[i] + 1] = 0;
    loglikelihood_y[i, y[i] + 1] = 0;
  }

  for (e in 1:Nedges) {
    int parent = edges[e, 1];
    int child = edges[e, 2];
    real t = edge_lengths[e];

    matrix[2,2] P1 = matrix_exp(Q1 * t);
    matrix[2,2] P2 = matrix_exp(Q2 * t);

    for (k in 1:2) {
      loglikelihood_x[parent, k] = log_sum_exp(to_vector(log(P1[k]) + loglikelihood_x[child]));
      loglikelihood_y[parent, k] = log_sum_exp(to_vector(log(P2[k]) + loglikelihood_y[child]));
    }
  }
}

model {
  rates ~ lognormal(-1, 0.5);
  pi_1 ~ dirichlet(rep_vector(2, 2));
  pi_2 ~ dirichlet(rep_vector(2, 2));

  target += log_sum_exp(loglikelihood_x[root_node] + log(pi_1)');
  target += log_sum_exp(loglikelihood_y[root_node] + log(pi_2)');
}

