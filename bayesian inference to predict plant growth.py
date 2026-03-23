import pymc as pm

# Taking draws from a normal distribution
seed = 42
x_dist = pm.Normal.dist(shape=(100, 3))
x_data = pm.draw(x_dist, random_seed=seed)

# Define coordinate values for all dimensions of the data
coords={
 "trial": range(100),
 "features": ["sunlight hours", "water amount", "soil nitrogen"],
}

# Define generative model
with pm.Model(coords=coords) as generative_model:
   x = pm.Data("x", x_data, dims=["trial", "features"])

   # Model parameters
   betas = pm.Normal("betas", dims="features")
   sigma = pm.HalfNormal("sigma")

   # Linear model
   mu = x @ betas

   # Likelihood
   # Assuming we measure deviation of each plant from baseline
   plant_growth = pm.Normal("plant growth", mu, sigma, dims="trial")


# Generating data from model by fixing parameters
fixed_parameters = {
 "betas": [5, 20, 2],
 "sigma": 0.5,
}
with pm.do(generative_model, fixed_parameters) as synthetic_model:
   idata = pm.sample_prior_predictive(random_seed=seed) # Sample from prior predictive distribution.
   synthetic_y = idata.prior["plant growth"].sel(draw=0, chain=0)


# Infer parameters conditioned on observed data
with pm.observe(generative_model, {"plant growth": synthetic_y}) as inference_model:
   idata = pm.sample(random_seed=seed)

   summary = pm.stats.summary(idata, var_names=["betas", "sigma"])
   print(summary)

   # Simulate new data conditioned on inferred parameters
new_x_data = pm.draw(
    pm.Normal.dist(shape=(3, 3)),
    random_seed=seed,
)
new_coords = coords | {"trial": [0, 1, 2]}

with inference_model:
    pm.set_data({"x": new_x_data}, coords=new_coords)
    pm.sample_posterior_predictive(
        idata,
        predictions=True,
        extend_inferencedata=True,
        random_seed=seed,
    )

print(pm.stats.summary(idata.predictions, kind="stats"))

# Simulate new data, under a scenario where the first beta is zero
with pm.do(
    inference_model,
    {inference_model["betas"]: inference_model["betas"] * [0, 1, 1]},
) as plant_growth_model:
    new_predictions = pm.sample_posterior_predictive(
        idata,
        predictions=True,
        random_seed=seed,
    )

print(pm.stats.summary(new_predictions, kind="stats"))