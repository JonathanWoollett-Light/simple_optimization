use itertools::izip;
use rand::distributions::uniform::SampleUniform;
use std::{
    f64,
    ops::{AddAssign, Div, Range, Sub},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use crate::util::{poll, Polling};

/// [Grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)
///
/// Evaluate all combinations of values from the 3 values where:
/// - Value 1 covers `10` values at equal intervals from `0..10` (`0,1,2,3,4,5,6,7,8,9`).
/// - Value 2 covers `11` values at equal intervals from `5..15`.
/// - Value 3 covers `12` values at equal intervals from `10..20`.
///
/// Printing progress every `10ms` and exiting early if a value is found which is less than or equal to `15.`.
/// ```
/// use std::{sync::Arc,time::Duration};
/// use simple_optimization::{grid_search, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 { list.iter().sum() }
/// let best = grid_search(
///     [0f64..10f64, 5f64..15f64, 10f64..20f64], // Value ranges.
///     simple_function, // Evaluation function.
///     None, //  No additional evaluation data.
///     // Polling every `10ms`, printing progress (`true`), exiting early if `15.` or less is reached, and not printing thread execution data (`false`).
///     Some(Polling { poll_rate: Duration::from_millis(5), printing: true, early_exit_minimum: Some(15.), thread_execution_reporting: false }),
///     // Take `10` samples along range `0` (`0..10`), `11` along range `1` (`5..15`)
///     //  and `12` along range `2` (`10..20`).
///     // In total taking `10*11*12=1320` samples.
///     [10,11,12],
/// );
/// assert_eq!(simple_function(&best, None), 15.);
/// ```
/// Due to specific design the `threads` parameter is excluded for now.
pub fn grid_search<
    A: 'static + Send + Sync,
    T: 'static
        + Copy
        + Send
        + Sync
        + Default
        + SampleUniform
        + PartialOrd
        + AddAssign
        + Sub<Output = T>
        + Div<Output = T>
        + num::FromPrimitive,
    const N: usize,
>(
    // Generics
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<Polling>,
    // Specifics
    points: [u64; N],
) -> [T; N] {
    // Compute step sizes
    let mut steps = [Default::default(); N];
    for (r, k, s) in izip!(ranges.iter(), points.iter(), steps.iter_mut()) {
        *s = (r.end - r.start) / T::from_u64(*k).unwrap();
    }

    // Compute point values
    let point_values: Vec<Vec<T>> = izip!(ranges.iter(), points.iter(), steps.iter())
        .map(|(r, k, s)| {
            (0..*k)
                .scan(r.start, |state, _| {
                    // Do this so the first value is `r.start` instead of `r.start+s` and the last value is `r.end-s` instead of r.end`.
                    let prev_state = *state;
                    *state += *s;
                    Some(prev_state)
                })
                .collect()
        })
        .collect();

    // Search points
    let mut start = [Default::default(); N];
    for (s, p) in start.iter_mut().zip(point_values.iter()) {
        *s = p[0];
    }
    let (_, params) = thread_search(f, evaluation_data, polling, &point_values, start);
    return params;

    fn thread_search<
        A: 'static + Send + Sync,
        T: 'static
            + Copy
            + Send
            + Sync
            + Default
            + SampleUniform
            + PartialOrd
            + AddAssign
            + Sub<Output = T>
            + Div<Output = T>
            + num::FromPrimitive,
        const N: usize,
    >(
        // Generics
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        polling: Option<Polling>,
        // Specifics
        point_values: &Vec<Vec<T>>,
        mut point: [T; N],
    ) -> (f64, [T; N]) {
        // Could just `assert!(N>0)` and not handle it, but this handles it fine.
        if 0 == point_values.len() {
            return (f(&point, evaluation_data), point);
        }

        let thread_exit = Arc::new(AtomicBool::new(false));
        // (handles,counters)
        let (handles, links): (Vec<_>, Vec<_>) = point_values[0]
            .iter()
            .map(|p_value| {
                point[0] = *p_value;
                let point_values_clone = point_values.clone();
                let counter = Arc::new(AtomicU64::new(0));
                let thread_best = Arc::new(Mutex::new(f64::MAX));
                let thread_execution_position = Arc::new(AtomicU8::new(0));
                let thread_execution_time = Arc::new([
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
                            &point_values_clone,
                            f,
                            evaluation_data_clone,
                            counter_clone,
                            thread_best_clone,
                            thread_exit_clone,
                            thread_execution_position_clone,
                            thread_execution_time_clone,
                            // Specifics
                            point,
                            1,
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
            let iterations = point_values.iter().map(|pvs| pvs.len() as u64).product();
            poll(
                poll_data,
                counters,
                0,
                iterations,
                thread_bests,
                thread_exit,
                thread_execution_positions,
                thread_execution_times,
            );
        }

        let joins: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let (value, params) =
            joins
                .into_iter()
                .fold((f64::MAX, [Default::default(); N]), |(bv, bp), (v, p)| {
                    if v < bv {
                        (v, p)
                    } else {
                        (bv, bp)
                    }
                });
        return (value, params);
    }
    fn search<
        A: 'static + Send + Sync,
        T: 'static
            + Copy
            + Send
            + Sync
            + Default
            + SampleUniform
            + PartialOrd
            + AddAssign
            + Sub<Output = T>
            + Div<Output = T>
            + num::FromPrimitive,
        const N: usize,
    >(
        // Generics
        point_values: &Vec<Vec<T>>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        counter: Arc<AtomicU64>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
        thread_execution_position: Arc<AtomicU8>,
        thread_execution_times: Arc<[Mutex<(Duration, u64)>; 2]>,
        // Specifics
        mut point: [T; N],
        index: usize,
    ) -> (f64, [T; N]) {
        if index == point_values.len() {
            // panic!("hit here");
            counter.fetch_add(1, Ordering::SeqCst);
            return (f(&point, evaluation_data), point);
        }

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for p_value in point_values[index].iter() {
            point[index] = *p_value;
            let (value, params) = search(
                point_values,
                f,
                evaluation_data.clone(),
                counter.clone(),
                best.clone(),
                thread_exit.clone(),
                thread_execution_position.clone(),
                thread_execution_times.clone(),
                point,
                index + 1,
            );
            if value < best_value {
                best_value = value;
                best_params = params;
                *best.lock().unwrap() = best_value;
            }
            if thread_exit.load(Ordering::SeqCst) {
                break;
            }
        }
        return (best_value, best_params);
    }
}
