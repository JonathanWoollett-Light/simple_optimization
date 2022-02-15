use itertools::{izip, Itertools};
use rand::distributions::uniform::SampleUniform;
use std::{
    convert::TryInto,
    f64, fmt,
    ops::{Add, AddAssign, Div, Mul, Range, Sub},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use crate::util::{poll, Polling};

/// Castes all given ranges to `f64` values and calls [`grid_search()`].
/// ```
/// use std::{sync::Arc,time::Duration};
/// use simple_optimization::{grid_search, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 { list.iter().sum() }
/// let best = grid_search!(
///     (0f64..10f64, 5u32..15u32, 10i16..20i16), // Value ranges.
///     simple_function, // Evaluation function.
///     None, //  No additional evaluation data.
///     // Polling every `10ms`, printing progress (`true`), exiting early if `15.` or less is reached, and not printing thread execution data (`false`).
///     Some(Polling { poll_rate: Duration::from_millis(5), printing: true, early_exit_minimum: Some(15.), thread_execution_reporting: false }),
///     None, // We don't specify the number of threads.
///     // Take `10` samples along range `0` (`0..10`), `11` along range `1` (`5..15`)
///     //  and `12` along range `2` (`10..20`).
///     // In total taking `10*11*12=1320` samples.
///     [10,11,12],
/// );
/// assert_eq!(simple_function(&best, None), 15.);
/// ```
/// Due to specific design the `threads` parameter is excluded for now.
#[macro_export]
macro_rules! grid_search {
    (
        // Generic
        ($($x:expr),*),
        $f: expr,
        $evaluation_data: expr,
        $polling: expr,
        $threads: expr,
        // Specific
        $points: expr,
    ) => {
        {
            use num::ToPrimitive;
            let mut ranges = [
                $(
                    $x.start.to_f64().unwrap()..$x.end.to_f64().unwrap(),
                )*
            ];
            simple_optimization::grid_search(
                ranges,
                $f,
                $evaluation_data,
                $polling,
                $threads,
                $points,
            )
        }
    };
}

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
///     None, // We don't specify the number of threads.
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
        + fmt::Debug
        + SampleUniform
        + PartialOrd
        + AddAssign
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + num::FromPrimitive,
    const N: usize,
>(
    // Generics
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<Polling>,
    threads: Option<usize>,
    // Specifics
    points: [u64; N],
) -> [T; N] {
    // Gets cpu number
    let cpus = crate::cpus!(threads);
    // 1 cpu is used for polling (this one), so we have -1 cpus for searching.
    let search_cpus = cpus - 1;
    // Computes points per thread
    let mut remainder = [Default::default(); N];
    let mut per = [Default::default(); N];
    for i in 0..N {
        remainder[i] = points[i] % search_cpus as u64;
        per[i] = std::cmp::max(points[i] / search_cpus as u64, 1);
    }

    // println!("remainder: {:?}, per: {:?}",remainder,per);

    // Points ranges for remainder
    // ---------------------------------------
    let remainder_ranges: [Range<u64>; N] = remainder
        .iter()
        .map(|&r| 0..r)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // Points ranges per thread
    // ---------------------------------------
    // If at any point the threads need to evaluate more than 1 value.
    let some_thread_work = per.iter().any(|&x| x > 1);

    // We effectively fold over our threads for each point range
    let per_ranges_opt: Option<Vec<[Range<u64>; N]>> = some_thread_work.then(|| {
        let mut offset = [Default::default(); N];
        // Initial offset is after remainder
        for i in 0..N {
            offset[i] = remainder_ranges[i].end;
        }

        (0..search_cpus)
            .map(|_| {
                (0..N)
                    .map(|i| {
                        let new = offset[i]..offset[i] + per[i];
                        offset[i] = new.end;
                        new
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    });

    // println!("remainder_ranges: {:?}, per_ranges_opt: {:?}",remainder_ranges,per_ranges_opt);

    // Checks ranges
    // ---------------------------------------
    // Number of evaluations all the threads do.
    let mut iterations = 0;
    // Number of evaluations the remainder does.
    let mut remainder = 0;
    for i in 0..N {
        // Gets points covered by remainder
        let remainder_point_sum = remainder_ranges[i].end - remainder_ranges[i].start;
        remainder += remainder_point_sum;
        // Gets points covered by threads
        let point_sum = per_ranges_opt.as_ref().map_or(0, |per_ranges| {
            per_ranges
                .iter()
                .fold(0, |acc, x| acc + x[i].end - x[i].start)
        });
        iterations += point_sum;
        // Checks sum
        assert_eq!(
            remainder_point_sum + point_sum,
            points[i],
            "remainder: {:?}, threads: {:?}",
            remainder_ranges,
            per_ranges_opt
        );
    }

    // Compute step sizes
    // ---------------------------------------
    let mut steps = [Default::default(); N];
    for (r, k, s) in izip!(ranges.iter(), points.iter(), steps.iter_mut()) {
        *s = (r.end - r.start) / T::from_u64(*k).unwrap();
    }

    // Covers remainder section
    // ---------------------------------------
    let ranges_arc = Arc::new(ranges);
    let (best_value, mut best_params) = search(
        // Generics
        ranges_arc.clone(),
        f,
        evaluation_data.clone(),
        // Since we are doing this on the same thread, we don't need to use these
        Arc::new(AtomicU64::new(Default::default())),
        Arc::new(Mutex::new(Default::default())),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicU8::new(0)),
        Arc::new([]),
        // Specifics
        remainder_ranges,
        steps,
    );

    // println!("completed remainder: {}",best_value);

    // Threads
    // ---------------------------------------

    if let Some(per_ranges) = per_ranges_opt {
        let thread_exit = Arc::new(AtomicBool::new(false));
        let (handles, links): (Vec<_>, Vec<_>) = (0..search_cpus)
            .zip(per_ranges.into_iter())
            .map(|(_, per_ranges)| {
                let ranges_clone = ranges_arc.clone();
                let counter = Arc::new(AtomicU64::new(0));
                let thread_best = Arc::new(Mutex::new(f64::MAX));
                let thread_execution_position = Arc::new(AtomicU8::new(0));
                let thread_execution_time = Arc::new([]);

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
                            per_ranges,
                            steps,
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

        // Joins all handles and folds across extracting the best value and best points.
        let (new_best_value, new_best_params) = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .fold((best_value, best_params), |(bv, bp), (v, p)| {
                if v < bv {
                    (v, p)
                } else {
                    (bv, bp)
                }
            });
        // If the best value from threads is better than the value from remainder
        if new_best_value < best_value {
            best_params = new_best_params
        }
    }

    return best_params;

    fn search<
        A: 'static + Send + Sync,
        T: 'static
            + Copy
            + Send
            + Sync
            + Default
            + fmt::Debug
            + SampleUniform
            + PartialOrd
            + AddAssign
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Mul<Output = T>
            + num::FromPrimitive,
        const N: usize,
    >(
        // Generics
        ranges: Arc<[Range<T>; N]>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        counter: Arc<AtomicU64>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
        _thread_execution_position: Arc<AtomicU8>,
        _thread_execution_times: Arc<[Mutex<(Duration, u64)>; 0]>,
        // Specifics
        point_ranges: [Range<u64>; N],
        steps: [T; N],
    ) -> (f64, [T; N]) {
        let (mut best_value, mut best_points) = (f64::MAX, [Default::default(); N]);

        let mut start_point = [Default::default(); N];
        for i in 0..N {
            start_point[i] = ranges[i].start;
        }
        // println!("start_point: {:?}",start_point);

        for cartesian_product in point_ranges
            .iter()
            .map(|r| r.clone())
            .multi_cartesian_product()
        {
            // Gets new point
            let mut point = start_point;

            // print!("[");
            for i in 0..N {
                // print!("{:?}*{:?}=",steps[i],T::from_u64(cartesian_product[i]).unwrap());
                point[i] += steps[i] * T::from_u64(cartesian_product[i]).unwrap();
                // print!("{:?},",point[i]);
            }
            // println!("] = {:?}",point);
            let new = f(&point, evaluation_data.clone());
            // println!("{:?} -> {:?}",point,new);
            if new < best_value {
                best_value = new;
                best_points = point;
                *best.lock().unwrap() = best_value;
            }
            counter.fetch_add(1, Ordering::SeqCst);
            // Checks early exit
            if thread_exit.load(Ordering::SeqCst) {
                break;
            }
        }

        (best_value, best_points)
    }
}
