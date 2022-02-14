// TODO Avoid stupid long repeated trait requirements for `T`.

//! Optimization algorithms.
//!
//! ## Crate status
//!
//! I made this since alternatives in the Rust ecosystem seemed awkward. I'm just some random dude, don't expect too much.
//!
//! Currently looking into adding macros `random_search!`, `grid_search!`, etc.
//!
//! These will allow you to optimize a tuple of ranges instead of an array, therefore allowing varying types.
//! This is likely to come before gaussian optimization.
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
//!     // Polling data, e.g. how often (if at all) you want to print progress, see `Polling` struct docs for more info.
//!     polling: Option<Polling>,
//!     // The number of threads you want to use, leaving this as `None` uses maximum (and is probably best). If set it must be >=2.
//!     threads: Option<usize>,
//!     // ...
//! ) -> [T;N] { /* ... */}
//! ```
//! The typical use case will be run with `--nocapture` (without this progress logging will not print), e.g.:
//! - `cargo run --nocapture`
//! - `cargo run --release -- --nocapture`
//! - `cargo test your_test --release -- --nocapture`
//!
//! The most thorough output of progress might look like:
//! ```ignore
//!  2300
//!   565 (24.57%) 00:00:11 / 00:00:47 [25.600657363049734] { [563.0ns, 561.3ms, 125.0ns, 110.0ns] [2.0µs, 361.8ms, 374.0ns, 405.0ns] [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] }
//! ```
//! This output describes:
//!
//! - The total number of iterations `2300`.
//! - The total number of completed iterations `565`.
//! - The percent of iterations completed `(24.57%)` (`565/2300=0.2457...`).
//! - The time running `00:00:11` (`hh:mm:ss`).
//! - The estimated time remaining `00:00:47` (`hh:mm:ss`).
//! - The current best value `[25.600657363049734]`.
//! - The most recently measured times between execution positions (effectively time taken for thread to go from some line, to another line (defined specifically with `update_execution_position` in the code) `[563.0ns, 561.3ms, 125.0ns, 110.0ns]`.
//! - The averages times between execution positions (this is average across entire runtime rather than since last measured) `[2.0µs, 361.8ms, 374.0ns, 405.0ns]`.
//! - The execution positions of threads (`0` is when a thread is completed, rest represent a thread having hit some line, which triggered this setting, but yet to hit next line which changes it, effectively being between 2 positions) (`[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`). What these specifically refer to in code varies between functions.
//!
//! The last 3 of these I wouldn't expect you would ever use. But I use them for debugging this library and I think they could possibly in some rare circumstance be useful to you (so no harm having them as an option, well, only a few microseconds of harm).

mod grid_search;
mod random_search;
mod simulated_annealing;
mod util;

#[macro_export]
#[doc(hidden)]
macro_rules! cpus {
    ($threads:expr) => {{
        // If `Some(n)` return `n` else `num_cpus::get()`.
        let cpus = $threads.unwrap_or_else(num_cpus::get) as u64;
        // Either way it must be at least 2.
        assert!(
            cpus >= 2,
            "Due to the fundamentally multi-threaded design, we need at least 2 threads"
        );
        cpus
    }};
}

pub use grid_search::*;
pub use random_search::*;
pub use simulated_annealing::*;
pub use util::Polling;
