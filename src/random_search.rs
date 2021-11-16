use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};

use std::{
    f64,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::util::{poll, update_execution_position, Polling};


/// Castes all given ranges to `f64` values and calls `random_search`.
/// ```
/// use std::sync::Arc;
/// use simple_optimization::{random_search_m, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc::<()>>) -> f64 { list.iter().sum() }
/// let best = random_search_m!(
///     (0f64..10f64, 5u32..15u32, 10i16..20i16), // Value ranges.
///     simple_function, // Evaluation function.
///     None, // No additional evaluation data.
///     // By using `new` this defaults to polling every `10ms`, we also print progress `true` and exit early if `19.` or less is reached.
///     Some(Polling::new(true,Some(19.))),
///     None,
///     1000, // Take `1000` samples (split between threads, so each thread only takes `1000/n` samples).
/// );
/// assert!(simple_function(&best, None) < 19.);
/// ```
#[macro_export]
macro_rules! random_search_m {
    (
        // Generic
        ($($x:expr),*),
        $f: expr,
        $evaluation_data: expr,
        $polling: expr,
        $threads: expr,
        // Specific
        $iterations: expr,
    ) => {
        {
            use num::ToPrimitive;
            let mut ranges = [
                $(
                    $x.start.to_f64().unwrap()..$x.end.to_f64().unwrap(),
                )*
            ];
            simple_optimization::random_search(
                ranges,
                $f,
                $evaluation_data,
                $polling,
                $threads,
                $iterations
            )
        }
    };
}

/// [Random search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Random_search)
///
/// Randomly pick parameters for `simple_function` in the ranges `0..5`, `5..15`, and `10..20` and return the parameters which produce the minimum result from `simple_function` out of `10,000` samples, printing progress every `10ms`, and exiting early if a value is found which is less than or equal to `19.`.
/// ```
/// use std::sync::Arc;
/// use simple_optimization::{random_search, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc::<()>>) -> f64 { list.iter().sum() }
/// let best = random_search(
///     [0f64..10f64, 5f64..15f64, 10f64..20f64], // Value ranges.
///     simple_function, // Evaluation function.
///     None, // No additional evaluation data.
///     // By using `new` this defaults to polling every `10ms`, we also print progress `true` and exit early if `19.` or less is reached.
///     Some(Polling::new(true,Some(19.))),
///     None,
///     1000, // Take `1000` samples (split between threads, so each thread only takes `1000/n` samples).
/// );
/// assert!(simple_function(&best, None) < 19.);
/// ```
pub fn random_search<
    A: 'static + Send + Sync,
    T: 'static + Copy + Send + Sync + Default + SampleUniform + PartialOrd,
    const N: usize,
>(
    // Generics
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<Polling>,
    threads: Option<usize>,
    // Specifics
    iterations: u64,
) -> [T; N] {
    // Gets cpu data
    let cpus = if let Some(given_threads) = threads {
        assert!(
            given_threads >= 2,
            "Due to fundamentally multi-threaded design, need at least 2 threads"
        );
        given_threads as u64
    } else {
        num_cpus::get() as u64
    };
    let search_cpus = cpus - 1; // 1 cpu is used for polling, this one.

    let remainder = iterations % search_cpus;
    let per = iterations / search_cpus;

    let ranges_arc = Arc::new(ranges);

    let (best_value, best_params) = search(
        // Generics
        ranges_arc.clone(),
        f,
        evaluation_data.clone(),
        // Since we are doing this on the same thread, we don't need to use these
        Arc::new(AtomicU64::new(Default::default())),
        Arc::new(Mutex::new(Default::default())),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicU8::new(0)),
        Arc::new([
            Mutex::new((Duration::new(0, 0), 0)),
            Mutex::new((Duration::new(0, 0), 0)),
            Mutex::new((Duration::new(0, 0), 0)),
            Mutex::new((Duration::new(0, 0), 0)),
        ]),
        // Specifics
        remainder,
    );

    let thread_exit = Arc::new(AtomicBool::new(false));
    // (handles,(counters,thread_bests))
    let (handles, links): (Vec<_>, Vec<_>) = (0..search_cpus)
        .map(|_| {
            let ranges_clone = ranges_arc.clone();
            let counter = Arc::new(AtomicU64::new(0));
            let thread_best = Arc::new(Mutex::new(f64::MAX));
            let thread_execution_position = Arc::new(AtomicU8::new(0));
            let thread_execution_time = Arc::new([
                Mutex::new((Duration::new(0, 0), 0)),
                Mutex::new((Duration::new(0, 0), 0)),
                Mutex::new((Duration::new(0, 0), 0)),
                Mutex::new((Duration::new(0, 0), 0)),
            ]);

            let counter_clone = counter.clone();
            let thread_best_clone = thread_best.clone();
            let thread_exit_clone = thread_exit.clone();
            let evaluation_data_clone = evaluation_data.clone();
            let thread_execution_position_clone = thread_execution_position.clone();
            let thread_execution_time_clone = thread_execution_time.clone();
            (
                thread::spawn(move || {
                    search(
                        // Generics
                        ranges_clone,
                        f,
                        evaluation_data_clone,
                        counter_clone,
                        thread_best_clone,
                        thread_exit_clone,
                        thread_execution_position_clone,
                        thread_execution_time_clone,
                        // Specifics
                        per,
                    )
                }),
                (
                    counter,
                    (
                        thread_best,
                        (thread_execution_position, thread_execution_time),
                    ),
                ),
            )
        })
        .unzip();
    let (counters, links): (Vec<Arc<AtomicU64>>, Vec<_>) = links.into_iter().unzip();
    let (thread_bests, links): (Vec<Arc<Mutex<f64>>>, Vec<_>) = links.into_iter().unzip();
    let (thread_execution_positions, thread_execution_times) = links.into_iter().unzip();

    if let Some(poll_data) = polling {
        poll(
            poll_data,
            counters,
            remainder,
            iterations,
            thread_bests,
            thread_exit,
            thread_execution_positions,
            thread_execution_times,
        );
    }

    let joins: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let (_, best_params) = joins
        .into_iter()
        .fold((best_value, best_params), |(bv, bp), (v, p)| {
            if v < bv {
                (v, p)
            } else {
                (bv, bp)
            }
        });

    return best_params;

    fn search<
        A: 'static + Send + Sync,
        T: 'static + Copy + Send + Sync + Default + SampleUniform + PartialOrd,
        const N: usize,
    >(
        // Generics
        ranges: Arc<[Range<T>; N]>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        counter: Arc<AtomicU64>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
        thread_execution_position: Arc<AtomicU8>,
        thread_execution_times: Arc<[Mutex<(Duration, u64)>; 4]>,
        // Specifics
        iterations: u64,
    ) -> (f64, [T; N]) {
        let mut execution_position_timer = Instant::now();
        let mut rng = thread_rng();
        let mut params = [Default::default(); N];

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for _ in 0..iterations {
            // Gen random values
            for (range, param) in ranges.iter().zip(params.iter_mut()) {
                *param = rng.gen_range(range.clone());
            }

            // Update execution position
            execution_position_timer = update_execution_position(
                1,
                execution_position_timer,
                &thread_execution_position,
                &thread_execution_times,
            );

            // Run function
            let new_value = f(&params, evaluation_data.clone());

            // Update execution position
            execution_position_timer = update_execution_position(
                2,
                execution_position_timer,
                &thread_execution_position,
                &thread_execution_times,
            );

            // Check best
            if new_value < best_value {
                best_value = new_value;
                best_params = params;
                *best.lock().unwrap() = best_value;
            }

            // Update execution position
            execution_position_timer = update_execution_position(
                3,
                execution_position_timer,
                &thread_execution_position,
                &thread_execution_times,
            );

            counter.fetch_add(1, Ordering::SeqCst);

            // Update execution position
            execution_position_timer = update_execution_position(
                4,
                execution_position_timer,
                &thread_execution_position,
                &thread_execution_times,
            );

            if thread_exit.load(Ordering::SeqCst) {
                break;
            }
        }
        // Update execution position
        // 0 represents ended state
        thread_execution_position.store(0, Ordering::SeqCst);
        return (best_value, best_params);
    }
}
