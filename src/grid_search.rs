use itertools::izip;
use rand::distributions::uniform::SampleUniform;
use std::{
    f64,
    ops::{AddAssign, Div, Range, Sub},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread,
};

use crate::util::poll;

/// Grid search
///
/// ```
/// use simple_optimization::grid_search;
/// use std::sync::Arc;
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 { list.iter().sum() }
/// let best = grid_search(
///     [10,10,10],
///     [0f64..10f64, 5f64..15f64, 10f64..20f64],
///     simple_function,
///     None,
///     Some(10),
///     Some(15.)
/// );
/// assert_eq!(simple_function(&best, None), 15.);
/// ```
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
    points: [u32; N],
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<u64>,
    early_exit_minimum: Option<f64>,
) -> [T; N] {
    // Compute step sizes
    let mut steps = [Default::default(); N];
    for (r, k, s) in izip!(ranges.iter(), points.iter(), steps.iter_mut()) {
        *s = (r.end - r.start) / T::from_u32(*k).unwrap();
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
    let (_, params) = thread_search(
        &point_values,
        f,
        evaluation_data,
        start,
        polling,
        early_exit_minimum,
    );
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
        point_values: &Vec<Vec<T>>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        mut point: [T; N],
        polling: Option<u64>,
        early_exit_minimum: Option<f64>,
    ) -> (f64, [T; N]) {
        // Could just `assert!(N>0)` and not handle it, but this handles it fine.
        if 0 == point_values.len() {
            return (f(&point, evaluation_data), point);
        }

        let thread_exit = Arc::new(AtomicBool::new(false));
        // (handles,counters)
        let (handles, links): (Vec<_>, Vec<(Arc<AtomicU32>, Arc<Mutex<f64>>)>) = point_values[0]
            .iter()
            .map(|p_value| {
                point[0] = *p_value;
                let point_values_clone = point_values.clone();
                let counter = Arc::new(AtomicU32::new(0));
                let thread_best = Arc::new(Mutex::new(f64::MAX));

                let counter_clone = counter.clone();
                let thread_best_clone = thread_best.clone();
                let thread_exit_clone = thread_exit.clone();
                let evaluation_data_clone = evaluation_data.clone();
                (
                    thread::spawn(move || {
                        search(
                            &point_values_clone,
                            f,
                            evaluation_data_clone,
                            point,
                            1,
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
            let iterations = point_values.iter().map(|pvs| pvs.len() as u32).product();
            poll(
                poll_rate,
                counters,
                0,
                iterations,
                early_exit_minimum,
                thread_bests,
                thread_exit,
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
        point_values: &Vec<Vec<T>>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        mut point: [T; N],
        index: usize,
        counter: Arc<AtomicU32>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
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
                point,
                index + 1,
                counter.clone(),
                best.clone(),
                thread_exit.clone(),
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
