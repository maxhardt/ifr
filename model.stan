// saved as model.stan
data {
  int<lower=0> N;
  int<lower=0> D[N];
  int<lower=0> P[N];
  int<lower=0> T[N];
  int<lower=0> CC[N];
}
parameters {
  vector[N] IFR_normal;
  vector[N] IR_normal;
  real<lower=1> bias;
  real<lower=0,upper=1> IFR_mu;
  real<lower=0,upper=1> IR_mu;
  real<lower=0> IFR_sigma;
  real<lower=0> IR_sigma;
}
transformed parameters {
  vector<lower=0,upper=1>[N] IFR = inv_logit(IFR_normal);
  vector<lower=0,upper=1>[N] IR = inv_logit(IR_normal);
  vector<lower=0,upper=1>[N] p = inv_logit(IR_normal + log(bias));
  real IFR_mu_normal = logit(IFR_mu);
  real IR_mu_normal = logit(IR_mu);
}
model {
  D ~ binomial(P, IFR .* IR);
  CC ~ binomial(T, p);
  IFR_normal ~ normal(IFR_mu_normal, IFR_sigma);
  IR_normal ~ normal(IR_mu_normal, IR_sigma);
  bias ~ normal(1, 0.4);
  IFR_mu ~ uniform(0, 1);
  IR_mu ~ uniform(0, 1);
  IFR_sigma ~ normal(0, 1);
  IR_sigma ~ normal(0, 10);
}
