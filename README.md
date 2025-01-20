# Symbolic Regression

Symbolic regression (SR) is a type of regression analysis that searches the space of mathematical expressions to find the model that best fits a given dataset, both in terms of accuracy and simplicity.\
In this solution I use a Genetic Programming algorithm to find the expected expression based on the MSE (Mean Square Error) value.

## Hyperparameters

The algorithm takes as input a `TrainingParameters` object, containing the parameters necessary:

- `x: numpy.ndarray`: an n×m array containing n samples with m variables each, representing the input variables.
- `y: numpy.ndarray`: an array containing m samples, representing the output.
- `num_generations: int`: the number of generations the algorithm will train for.
- `population_size: int`: the size of the population of each generation.
- `mse_tolerance: float`: the relative difference in MSE values that two individuals need to have to be compared based on tree length.
- `random_event_chance: float`: the chance, between 0.0 and 1.0, of any random event in the training to happen.

## Parent selection

The algorithm performs parent selection by taking two random samples in a pool of the best `population_size // 2` individuals.\
The two parents are selected randomly, with a distribution (that defines the probability of being selected) that depends on the `weighted_parent_selection` hyperparameter:

- if the value of the parameter is set to `False`, the distribution is a uniform distribution, and each individual has an equal chance to be selected.
- if the value of the parameter is set to `True`, the distribution is given by the normalized MSE values of the individuals (e.g.the MSE of each individual divided by the sum of the MSE values of the entire pool). This way, individuals with better MSE are more likely to be selected.

## Crossover

The algorithm performs crossover by taking a non-leaf subtree of one of the parents, and replacing a random node of the other parent.\
Taking a non-leaf subtree ensures that, on the long run, the size of the trees will keep increasing.

## Mutation

The algorithm performs mutation by randomly replacing half of the nodes of an individual, according to the following scheme:

- if the node is a leaf, it is replaced with another random leaf (e.g. a variable or a constant).
- if the node is an operator, it is replaced with an operator of the same arity (number of inputs). It also has a chance, defined by the `random_event_chance` parameter, of being replaced by a leaf, reducing the overall tree size of the individual.

## Fitness and sorting

The `Fitness` custom class is used to implement custom sorting of individuals.\
The individuals are generally sorted based on the MSE value, but there is also a chance to prefer an individual with a slightly higher MSE value.\
Specifically, if the ratio between their MSE values of two individuals is at most `1 + mse_tolerance`, they are sorted based on tree size instead.

## Hyperparameter selection

First, I performed a general analysis on the `num_generations` and `population_size` parameters.\
I trained the model on the `problem_0.npz` dataset, performing a grid search on the following values:

- `num_generations`: 100, 1000, 10000
- `population_size`: 10, 20, 50

The results are the following:

| MSE                   | `num_generations`: 100 | `num_generations`: 1000 | `num_generations`: 10000 |
|-----------------------|------------------------|-------------------------|--------------------------|
| `population_size`: 10 | 3.20                   | 3.39                    | 3.80                     |
| `population_size`: 20 | 1.99                   | 3.20                    | 2.14                     |
| `population_size`: 50 | 0.01                   | 1.85                    | 1.89                     |

As it is evident, a `population_size` of 50 yields the best results.

As for the `num_generations` parameter, all values have similar results, so I repeated the analysis, this time for 10 iterations and with values of 100, 500 and 1000.\
I excluded 10000 because it was overkill, and it also cost more in terms of time (45 minutes on average, upwards of 4h, for a single training cycle).

This new analysis provided the results below:

| MSE        | `num_generations`: 100 | `num_generations`: 500 | `num_generations`: 1000 |
|------------|------------------------|------------------------|-------------------------|
| Attempt 1  | 0.0107                 | 0.0107                 | 0.0107                  |
| Attempt 2  | 2.203e-5               | 0.0107                 | 0.0090                  |
| Attempt 3  | 0.0107                 | 0.0107                 | 0.0107                  |
| Attempt 4  | 0.0107                 | 0.0107                 | 0.0107                  |
| Attempt 5  | 0.0107                 | 0.0107                 | 0.0107                  |
| Attempt 6  | 0.0014                 | 0.0106                 | 0.0106                  |
| Attempt 7  | 0.0325                 | 2.3699e-5              | 1.4652e-7               |
| Attempt 8  | 0.0107                 | 0.0107                 | 0.0107                  |
| Attempt 9  | 0.0107                 | 0.0016                 | 0.2543                  |
| Attempt 10 | 0.0107                 | 0.0105                 | 0.0107                  |

Seeing as all three values produced generally similar results, I opted to train my models for 500 generations, in order to best balance accuracy and training time.



To select the values of the other hyperparameters, I also performed grid searches, one for each problem, on the following values:

- `mse_tolerance`: 0.0, 0.05, 0.1
- `random_event_chance`: 0.0, 0.25, 0.5, 0.75, 1.0

The best results were as following:

| MSE                         | `problem_1.npz` | `problem_2.npz` | `problem_3.npz` | `problem_4.npz` | `problem_5.npz` | `problem_6.npz` | `problem_7.npz` | `problem_8.npz` |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| `mse_tolerance`             | 0.0             | 0.0             | 0.1             | 0.0             | 0.1             | 0.05            | 0.05            | 0.05            |
| `random_event_chance`       | 0.75            | 0.25            | 0.0             | 0.25            | 0.0             | 0.25            | 0.0             | 0.5             |

# Final solutions

Finally, I trained the model 10 times for each problem, selecting the best solutions.
The final results are the following:

| Node                                                                                                                                                                                         | MSE               |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| sin(x0)                                                                                                                                                                                      | 7.1259e-34        |
| multiply(625, multiply(multiply(multiply(4, multiply(multiply(4, 0.577216), multiply(4, 0.577216))), x0), multiply(multiply(4, multiply(4, 0.577216)), multiply(4, multiply(4, 0.577216))))) | 19019678186349.03 |
| subtract(subtract(cosh(sinh(cosh(cos(cos(cosh(x0)))))), sinh(x1)), subtract(sinh(x1), cosh(x0)))                                                                                             | 188.4463          |
| add(add(cos(x1), exp(cos(x1))), add(add(cos(x1), cos(cos(x1))), add(cos(x1), add(cos(x1), exp(cos(x1))))))                                                                                   | 0.41458           |
| multiply(log10(sinh(sinh(x1))), sinh(tanh(tanh(0))))                                                                                                                                         | 5.5728e-18        |
| add(subtract(x1, x0), add(x1, cos(tanh(exp(subtract(subtract(x1, x0), x0))))))                                                                                                               | 0.5737            |
| exp(add(multiply(x1, x0), cosh(tanh(multiply(x1, x0)))))                                                                                                                                     | 331.0792          |
| add(add(add(power(x5, 5), power(x5, 5)), add(power(x5, 5), power(x5, 5))), add(power(x5, 5), -1))                                                                                            | 735292.5512       |