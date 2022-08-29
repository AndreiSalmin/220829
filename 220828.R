setwd("E:/R/Rworkshop")

install.packages("tidymodels")
install.packages("tidyverse")

library("rlang")
library("tidymodels")
library("tidyverse")
# url where the data is located
url <- "https://dax-cdn.cdn.appdomain.cloud/dax-airline/1.0.1/lax_to_jfk.tar.gz"

# download the file
download.file(url, destfile = "lax_to_jfk.tar.gz")

# untar the file so we can get the csv only
# if you run this on your local machine, then can remove tar = "internal" 
untar("lax_to_jfk.tar.gz", tar = "internal")

# read_csv only 
sub_airline <- read_csv("lax_to_jfk/lax_to_jfk.csv",
                        col_types = cols('DivDistance' = col_number(), 
                                         'DivArrDelay' = col_number()))

flight_delays <- sub_airline %>% 
  replace_na(list(CarrierDelay = 0,
                  WeatherDelay = 0,
                  NASDelay = 0,
                  SecurityDelay = 0,
                  LateAircraftDelay = 0)) %>%
  select(c(ArrDelayMinutes, DepDelayMinutes, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, DayOfWeek, Month))

set.seed(1234)
flight_split <- initial_split(flight_delays)
train_data <- training(flight_split)
test_data <- testing(flight_split)

# Pick linear regression
lm_spec <- linear_reg() %>%
  # Set engine
  set_engine(engine = "lm")

# Print the linear function
lm_spec

train_fit <- lm_spec %>% 
  fit(ArrDelayMinutes ~ DepDelayMinutes, data = train_data)

train_fit 


train_results <- train_fit %>%
  # Make the predictions and save the predicted values
  predict(new_data = train_data) %>%
  # Create a new column to save the true values
  mutate(truth = train_data$ArrDelayMinutes)

head(train_results)


test_results <- train_fit %>%
  # Make the predictions and save the predicted values
  predict(new_data = test_data) %>%
  # Create a new column to save the true values
  mutate(truth = test_data$ArrDelayMinutes)

head(test_results)

rmse(train_results, truth = truth,
     estimate = .pred)

flight_split2 <- initial_split(flight_delays, prop = 4/5)  # prop = 0.8 works as well
train_data2 <- training(flight_split2)
test_data2 <- testing(flight_split2)

rmse(test_results, truth = truth,estimate = .pred)

rsq(train_results, truth = truth,estimate = .pred)

rsq(test_results, truth = truth,estimate = .pred)

test_results %>%
  mutate(train = "testing") %>%
  bind_rows(train_results %>% mutate(train = "training")) %>%
  ggplot(aes(truth, .pred)) +
  geom_abline(lty = 2, color = "orange", 
              size = 1.5) +
  geom_point(color = '#006EA1', 
             alpha = 0.5) +
  facet_wrap(~train) +
  labs(x = "Truth", 
       y = "Predicted Arrival Delays (min)")

train_fit2 <- lm_spec %>% 
  fit(ArrDelayMinutes ~ DepDelayMinutes, 
      data = train_data2)
test_results2 <- train_fit2 %>%
  # Make the predictions and save the predicted values
  predict(new_data = test_data2) %>%
  # Create a new column to save the true values
  mutate(truth = test_data2$ArrDelayMinutes)
rsq(test_results2, truth = truth,
    estimate = .pred)

set.seed(1234)
cv_folds <- vfold_cv(train_data, v = 10)
results <- fit_resamples(lm_spec, 
                         ArrDelayMinutes ~ DepDelayMinutes,
                         resamples = cv_folds)

results %>% collect_metrics()

cv_folds_3 <- vfold_cv(train_data, v = 3)
results <- fit_resamples(
  lm_spec, 
  ArrDelayMinutes ~ DepDelayMinutes, 
  resamples = cv_folds_3)
results %>% collect_metrics()

ggplot(cars, aes(x = speed, y = dist)) + 
  geom_point() + 
  geom_hline(yintercept = mean(cars$dist), 
             col = "red") 
ggplot(cars, aes(x = speed, y = dist)) + 
  geom_point() + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, 8), 
              col = "red", se = FALSE) 
ggplot(cars, aes(x = speed, y = dist)) + 
  geom_point() + 
  geom_smooth(method = "lm", 
              formula = y ~ x, 
              col = "red", 
              se = FALSE) 
flight_recipe <-
  recipe(ArrDelayMinutes ~ ., data = train_data)

ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet")

ridge_wf <- workflow() %>%
  add_recipe(flight_recipe)

ridge_fit <- ridge_wf %>%
  add_model(ridge_spec) %>%
  fit(data = train_data)

install.packages("glmnet")
library("glmnet")

ridge_fit %>%
  extract_fit_parsnip() %>%
  tidy()

lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(flight_recipe)

lasso_fit <- lasso_wf %>%
  add_model(lasso_spec) %>%
  fit(data = train_data)

flight_recipe <-
  recipe(ArrDelayMinutes ~ ., data = train_data)

el_spec <- linear_reg(penalty = 0.5, mixture = 0.2) %>%
  set_engine("glmnet")

el_wf <- workflow() %>%
  add_recipe(flight_recipe)

el_fit <- el_wf %>%
  add_model(el_spec) %>%
  fit(data = train_data)

el_fit %>%
  pull_workflow_fit() %>%
  tidy()

tune_spec <- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(flight_recipe)

flight_cvfolds <- vfold_cv(train_data)

lambda_grid <- grid_regular(levels = 50,
                            penalty(range = c(-3, 0.3)))

lasso_grid <- tune_grid(
  lasso_wf %>% add_model(tune_spec), 
  resamples = flight_cvfolds, 
  grid = lambda_grid)

show_best(lasso_grid, metric = "rmse")

lasso_grid %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(penalty, mean)) +
  geom_line(size=1, color="red") +
  scale_x_log10() +
  ggtitle("RMSE")

tune_spec <- linear_reg(
  penalty = tune(), 
  mixture = 0) %>% 
  set_engine("glmnet")

ridge_grid <- tune_grid(ridge_wf %>% 
                          add_model(tune_spec), 
                        resamples = flight_cvfolds, 
                        grid = lambda_grid)

show_best(ridge_grid, metric = "rmse")
