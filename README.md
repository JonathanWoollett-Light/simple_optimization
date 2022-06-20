# simple_optimization

[![Crates.io](https://img.shields.io/crates/v/simple_optimization)](https://crates.io/crates/simple_optimization)
[![lib.rs.io](https://img.shields.io/crates/v/simple_optimization?color=blue&label=lib.rs)](https://lib.rs/crates/simple_optimization)
[![docs](https://img.shields.io/crates/v/simple_optimization?color=yellow&label=docs)](https://docs.rs/simple_optimization)

Some simple multi-threaded optimizers.

You could do a random search like:
```rust
use std::sync::Arc;
use simple_optimization::{random_search, Polling};
// Our evaluation function takes 3 `f64`s and no additional data `()`.
fn simple_function(list: &[f64; 3], _: Option<Arc::<()>>) -> f64 { list.iter().sum() }
let best = random_search!(
    (0f64..10f64, 5u32..15u32, 10i16..20i16), // Value ranges.
    simple_function, // Evaluation function.
    None, // No additional evaluation data.
    // Every `10ms` we print progress and exit early if `simple_function` return a value less than `19.`.
    Some(Polling::new(true,Some(19.))),
    None, // We don't specify the number of threads to run.
    1000, // Take `1000` samples.
);
assert!(simple_function(&best, None) < 19.);
```
Which during execution will give an output like:
```
1000
 500 (50.00%) 00:00:11 / 00:00:47 [25.600657363049734]
```
Representing:
```
<Total number of samples>
<Current number of samples> (<Percentage of samples taken>) <Time running> / <ETA> [<Current best value>]
```


### Support

Optimizer | Status
---|---
[Random search](https://en.wikipedia.org/wiki/Random_search)|✅
[Grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)|✅
[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)|✅
[Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)|WIP
[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)| [See my note here](https://github.com/JonathanWoollett-Light/cogent/blob/master/README.md#a-note)
[Genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)| No plans
[Ant colony optimization](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)| No plans
[Linear programming](https://en.wikipedia.org/wiki/Linear_programming)| No plans

I made this for my own use since the existing libraries I found felt awkward.
