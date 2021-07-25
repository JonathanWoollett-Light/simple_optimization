use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};

use std::{
    f64,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread,
};

use crate::util::poll;

/// Random search
///
/// Randomly pick parameters for `simple_function` in the ranges `0..1`, `1..2`, and `3..4` and return the parameters which produce the minimum result from `simple_function` out of 10,000 samples.
///
/// And every 10ms print progress.
/// ```
/// use simple_optimization::random_search;
/// use std::sync::Arc;
/// fn simple_function(list: &[f64; 3], _: Option<Arc::<()>>) -> f64 { list.iter().sum() }
/// let best = simple_optimization::random_search(
///     1000,
///     [0f64..10f64, 5f64..15f64, 10f64..20f64],
///     simple_function,
///     None,
///     None,
///     Some(19.)
/// );
// assert!(simple_function(&best, None) < 19.);
/// ```
pub fn random_search<
    A: 'static + Send + Sync,
    T: 'static + Copy + Send + Sync + Default + SampleUniform + PartialOrd,
    const N: usize,
>(
    iterations: u32,
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<u64>,
    early_exit_minimum: Option<f64>,
) -> [T; N] {
    // Gets cpu data
    let cpus = num_cpus::get() as u32;
    let remainder = iterations % cpus;
    let per = iterations / cpus;

    let ranges_arc = Arc::new(ranges);

    let (best_value, best_params) = search(
        remainder,
        ranges_arc.clone(),
        f,
        evaluation_data.clone(),
        // Since we are doing this on the same thread, we don't need to use these
        Arc::new(AtomicU32::new(Default::default())),
        Arc::new(Mutex::new(Default::default())),
        Arc::new(AtomicBool::new(false)),
    );

    let thread_exit = Arc::new(AtomicBool::new(false));
    // (handles,(counters,thread_bests))
    let (handles, links): (Vec<_>, Vec<(Arc<AtomicU32>, Arc<Mutex<f64>>)>) = (0..cpus)
        .map(|_| {
            let ranges_clone = ranges_arc.clone();
            let counter = Arc::new(AtomicU32::new(0));
            let thread_best = Arc::new(Mutex::new(f64::MAX));

            let counter_clone = counter.clone();
            let thread_best_clone = thread_best.clone();
            let thread_exit_clone = thread_exit.clone();
            let evaluation_data_clone = evaluation_data.clone();
            (
                thread::spawn(move || {
                    search(
                        per,
                        ranges_clone,
                        f,
                        evaluation_data_clone,
                        counter_clone,
                        thread_best_clone,
                        thread_exit_clone,
                    )
                }),
                (counter, thread_best),
            )
        })
        .unzip();
    let (counters, thread_bests): (Vec<Arc<AtomicU32>>, Vec<Arc<Mutex<f64>>>) =
        links.into_iter().unzip();

    if let Some(poll_rate) = polling {
        poll(
            poll_rate,
            counters,
            remainder,
            iterations,
            early_exit_minimum,
            thread_bests,
            thread_exit,
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
        iterations: u32,
        ranges: Arc<[Range<T>; N]>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        counter: Arc<AtomicU32>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
    ) -> (f64, [T; N]) {
        let mut rng = thread_rng();
        let mut params = [Default::default(); N];

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for _ in 0..iterations {
            // Gen random values
            for (range, param) in ranges.iter().zip(params.iter_mut()) {
                *param = rng.gen_range(range.clone());
            }
            // Run function
            let new_value = f(&params, evaluation_data.clone());
            // Check best
            if new_value < best_value {
                best_value = new_value;
                best_params = params;
                *best.lock().unwrap() = best_value;
            }
            counter.fetch_add(1, Ordering::SeqCst);
            if thread_exit.load(Ordering::SeqCst) {
                break;
            }
        }
        return (best_value, best_params);
    }
}
