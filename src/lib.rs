// TODO Avoid stupid long repeated trait requirements for `T`.

//! Optimization algorithms.
//!
//! ## Crate status
//! 
//! I made this since alternatives in the Rust ecosystem seemed awkward. I'm just some random dude, don't expect too much.
//!
//! ## Basic guide
//!
//! All the functions have the approximate form:
//! ```ignore
//! # use std::sync::Arc;
//! fn function<A,T,const N: usize>(
//!     // The values you want to optimise and there respective ranges.
//!     // E.g. Optimizing 2 `f32`s between 0..1 and 2..3 `[0f32..1f32, 2f32..3f32]`).
//!     ranges: [Range<T>; N], 
//!     // The function you want to optimize (loss/cost/whatever).
//!     // `&[T; N]` are the input parameters which the algorithm will adjust to
//!     //  minimize the function.
//!     // `Option<Arc<A>>` is how you can pass additional data you might want to use.
//!     f: fn(&[T; N], Option<Arc<A>>) -> f64,
//!     // The additional data for the evaluation function.
//!     evaluation_data: Option<Arc<A>>,
//!     // How often (if at all) you want to poll and print progress.
//!     polling: Option<u64>,
//!     // If this value is reached, the function exits immediately.
//!     // When this is `None` if a random search hit the optimum on its first guess 
//!     //  it would still continue to guess many more times (however many you set) 
//!     //  before returning.
//!     early_exit_minimum: Option<f64>,
//!     // ...
//! ) -> [T;N] { /* ... */}
//! ```
mod random_search;
mod grid_search;
mod simulated_annealing;
mod util;

pub use random_search::*;
pub use grid_search::*;
pub use simulated_annealing::*;