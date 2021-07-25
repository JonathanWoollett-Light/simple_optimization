// TODO Avoid stupid long repeated trait requirements for `T`.

pub mod grid_search;
pub mod random_search;
pub mod simulated_annealing;
mod util;

pub use grid_search::*;
pub use random_search::*;
pub use simulated_annealing::*;
