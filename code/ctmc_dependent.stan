functions {
  vector stationary_distribution(matrix Q) {
    int K = rows(Q);
    matrix[K, K] A = Q';
    vector[K] b = rep_vector(0.0, K);
    
    // Replace one row by sum-to-one constraint
    A[K] = rep_row_vector(1.0, K);
    b[K] = 1.0;

    return mdivide_left(A, b);  // solve A * pi = b
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

transformed data {
   array[N] int z;
   for (i in 1:N) {
      z[i] = 2* x[i] + y[i];
   }
}

parameters {
  real<lower=0> total_rate;
  simplex[8] rates;
}

transformed parameters {
  // Q matrices
  matrix[4, 4] Q;
  Q[1,1] = -rates[1] - rates[2];
  Q[1,2] = rates[1];
  Q[1,3] = rates[2];
  Q[1,4] = 0;
  Q[2,1] = rates[3];
  Q[2,2] = -rates[3] - rates[4];
  Q[2,3] = 0;
  Q[2,4] = rates[4];
  Q[3,1] = rates[5];
  Q[3,2] = 0;
  Q[3,3] = -rates[5] - rates[6];
  Q[3,4] = rates[6];
  Q[4,1] = 0;
  Q[4,2] = rates[7];
  Q[4,3] = rates[8];
  Q[4,4] = -rates[7] - rates[8];
  Q = total_rate * Q;

  vector[4] pi = stationary_distribution(Q);

  matrix[Nnodes, 4] loglikelihood;

  for (n in 1:Nnodes) {
    loglikelihood[n] = rep_row_vector(negative_infinity(), 4);
  }

  for (i in 1:N) {
    loglikelihood[i, z[i] + 1] = 0;
  }

  for (e in 1:Nedges) {
    int parent = edges[e, 1];
    int child = edges[e, 2];
    real t = edge_lengths[e];

    matrix[4,4] P = matrix_exp(Q * t);

    for (k in 1:4) {
      loglikelihood[parent, k] = log_sum_exp(to_vector(log(P[k]) + loglikelihood[child]));
    }
  }
}

model {
  total_rate ~ gamma(1, 1);
  rates ~ dirichlet(rep_vector(1, 8));

  target += log_sum_exp(loglikelihood[root_node] + log(pi)');
}

