use itertools::izip;
use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};
use rand_distr::{Distribution, Normal};

use std::{
    convert::TryInto,
    f64,
    ops::{Range, Sub},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::util::{poll, update_execution_position, Polling};

/// Cooling schedule for simulated annealing.
#[derive(Clone, Copy)]
pub enum CoolingSchedule {
    /// $ t_{n+1} = t \cdot \ln{\frac{\ln{2}}{s+1}} $
    Logarithmic,
    /// $ t_{n+1} = x \cdot t $
    Exponential(f64),
    /// $ t_n = \frac{t_1}{n} $
    Fast,
}
impl CoolingSchedule {
    fn decay(&self, t_start: f64, t_current: f64, step: u64) -> f64 {
        match self {
            Self::Logarithmic => t_current * (2f64).ln() / ((step + 1) as f64).ln(),
            Self::Exponential(x) => x * t_current,
            Self::Fast => t_start / step as f64,
        }
    }
    /// Given temperature start and temperature min, gives number of steps of decay which will occur
    ///  before temperature start decays to be less than temperature min, then causing program to exit.
    fn steps(&self, t_start: f64, t_min: f64) -> u64 {
        match self {
            Self::Logarithmic => (((2f64).ln() * t_start / t_min).exp() - 1f64).ceil() as u64,
            Self::Exponential(x) => ((t_min / t_start).log(*x)).ceil() as u64,
            Self::Fast => (t_start / t_min).ceil() as u64,
        }
    }
}

/// Castes all given ranges to `f64` values and calls [`simulated_annealing()`].
/// ```
/// use std::sync::Arc;
/// use simple_optimization::{simulated_annealing, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 {
///  list.iter().sum()
/// }
/// let best = simulated_annealing!(
///     (0f64..10f64, 5u32..15u32, 10i16..20i16), // Value ranges.
///     simple_function, // Evaluation function.
///     None, // No additional evaluation data.
///     // By using `new` this defaults to polling every `10ms`, we don't print progress `false` and exit early if `19.` or less is reached.
///     Some(Polling::new(false,Some(17.))),
///     None, // Don't specify the number of threads.
///     100., // Starting temperature is `100.`.
///     1., // Minimum temperature is `1.`.
///     simple_optimization::CoolingSchedule::Fast, // Use fast cooling schedule.
///     // Take `100` samples per temperature
///     // This is split between threads, so each thread only samples
///     //  `100/n` at each temperature.
///     100,
///     1., // Variance in sampling.
/// );
/// assert!(simple_function(&best, None) < 19.);
/// ```
#[macro_export]
macro_rules! simulated_annealing {
    (
        // Generic
        ($($x:expr),*),
        $f: expr,
        $evaluation_data: expr,
        $polling: expr,
        $threads: expr,
        // Specific
        $starting_temperature: expr,
        $minimum_temperature: expr,
        $cooling_schedule: expr,
        $samples_per_temperature: expr,
        $variance: expr,
    ) => {
        {
            use num::ToPrimitive;
            let mut ranges = [
                $(
                    $x.start.to_f64().unwrap()..$x.end.to_f64().unwrap(),
                )*
            ];
            simple_optimization::simulated_annealing(
                ranges,
                $f,
                $evaluation_data,
                $polling,
                $threads,
                $starting_temperature,
                $minimum_temperature,
                $cooling_schedule,
                $samples_per_temperature,
                $variance
            )
        }
    };
}

// TODO Multi-thread this
/// [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
///
/// Run simulated annealing starting at temperature `100.` decaying with a fast cooling schedule until reach a minimum temperature of `1.`, taking `100` samples at each temperature, with a variance in sampling of `1.`.
/// ```
/// use std::sync::Arc;
/// use simple_optimization::{simulated_annealing, Polling};
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 {
///  list.iter().sum()
/// }
/// let best = simulated_annealing(
///     [0f64..10f64, 5f64..15f64, 10f64..20f64], // Value ranges.
///     simple_function, // Evaluation function.
///     None, // No additional evaluation data.
///     // By using `new` this defaults to polling every `10ms`, we don't print progress `false` and exit early if `19.` or less is reached.
///     Some(Polling::new(false,Some(17.))),
///     None, // Don't specify the number of threads.
///     100., // Starting temperature is `100.`.
///     1., // Minimum temperature is `1.`.
///     simple_optimization::CoolingSchedule::Fast, // Use fast cooling schedule.
///     // Take `100` samples per temperature
///     // This is split between threads, so each thread only samples
///     //  `100/n` at each temperature.
///     100,
///     1., // Variance in sampling.
/// );
/// assert!(simple_function(&best, None) < 19.);
/// ```
pub fn simulated_annealing<
    A: 'static + Send + Sync,
    T: 'static
        + Copy
        + Send
        + Sync
        + Default
        + SampleUniform
        + PartialOrd
        + Sub<Output = T>
        + num::ToPrimitive
        + num::FromPrimitive,
    const N: usize,
>(
    // Generic
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    polling: Option<Polling>,
    threads: Option<usize>,
    // Specific
    starting_temperature: f64,
    minimum_temperature: f64,
    cooling_schedule: CoolingSchedule,
    samples_per_temperature: u64,
    variance: f64,
) -> [T; N] {
    let cpus = crate::cpus!(threads);

    let search_cpus = cpus - 1; // 1 cpu is used for polling, this one.

    let steps = cooling_schedule.steps(starting_temperature, minimum_temperature);
    let thread_exit = Arc::new(AtomicBool::new(false));
    let ranges_arc = Arc::new(ranges);

    let remainder = samples_per_temperature % search_cpus;
    let per = samples_per_temperature / search_cpus;

    let (best_value, best_params) = search(
        // Generics
        ranges_arc.clone(),
        f,
        evaluation_data.clone(),
        Arc::new(AtomicU64::new(Default::default())),
        Arc::new(Mutex::new(Default::default())),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicU8::new(0)),
        Arc::new([
            Mutex::new((Duration::new(0, 0), 0)),
            Mutex::new((Duration::new(0, 0), 0)),
            Mutex::new((Duration::new(0, 0), 0)),
        ]),
        // Specifics
        starting_temperature,
        minimum_temperature,
        cooling_schedule,
        remainder,
        variance,
    );

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
                        starting_temperature,
                        minimum_temperature,
                        cooling_schedule,
                        per,
                        variance,
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
            steps * remainder,
            steps * samples_per_temperature,
            thread_bests,
            thread_exit,
            thread_execution_positions,
            thread_execution_times,
        );
    }

    // Joins all handles and folds across extracting best value and best points.
    let (_, best_params) = handles.into_iter().map(|h| h.join().unwrap()).fold(
        (best_value, best_params),
        |(bv, bp), (v, p)| {
            if v < bv {
                (v, p)
            } else {
                (bv, bp)
            }
        },
    );

    return best_params;

    fn search<
        A: 'static + Send + Sync,
        T: 'static
            + Copy
            + Send
            + Sync
            + Default
            + SampleUniform
            + PartialOrd
            + Sub<Output = T>
            + num::ToPrimitive
            + num::FromPrimitive,
        const N: usize,
    >(
        // Generic
        ranges: Arc<[Range<T>; N]>,
        f: fn(&[T; N], Option<Arc<A>>) -> f64,
        evaluation_data: Option<Arc<A>>,
        counter: Arc<AtomicU64>,
        best: Arc<Mutex<f64>>,
        thread_exit: Arc<AtomicBool>,
        thread_execution_position: Arc<AtomicU8>,
        thread_execution_times: Arc<[Mutex<(Duration, u64)>; 3]>,
        // Specific
        starting_temperature: f64,
        minimum_temperature: f64,
        cooling_schedule: CoolingSchedule,
        samples_per_temperature: u64,
        variance: f64,
    ) -> (f64, [T; N]) {
        let mut execution_position_timer = Instant::now();
        let mut rng = thread_rng();
        // Get initial point
        let mut current_point = [Default::default(); N];
        for (p, r) in current_point.iter_mut().zip(ranges.iter()) {
            *p = rng.gen_range(r.clone());
        }
        let mut best_point = current_point;

        let mut current_value = f(&best_point, evaluation_data.clone());
        let mut best_value = current_value;

        // Gets ranges in f64
        // Since `Range` doesn't implement copy and array initialization will not clone,
        //  this bypasses it.
        let mut float_ranges: [Range<f64>; N] = vec![Default::default(); N].try_into().unwrap();
        for (float_range, range) in float_ranges.iter_mut().zip(ranges.iter()) {
            *float_range = range.start.to_f64().unwrap()..range.end.to_f64().unwrap();
        }
        // Variances scaled to the different ranges.
        let mut scaled_variances: [f64; N] = [Default::default(); N];
        for (scaled_variance, range) in scaled_variances.iter_mut().zip(float_ranges.iter()) {
            *scaled_variance = (range.end - range.start) * variance
        }

        let mut step = 1;
        let mut temperature = starting_temperature;
        // Iterate while starting temperature has yet to cool to the minimum temperature.
        while temperature >= minimum_temperature {
            // Distributions to sample from at this temperature.
            // `Normal::new(1.,1.).unwrap()` just replacement for `default()` since it doesn't implement trait.
            let mut distributions: [Normal<f64>; N] = [Normal::new(1., 1.).unwrap(); N];
            for (distribution, variance, point) in izip!(
                distributions.iter_mut(),
                scaled_variances.iter(),
                current_point.iter()
            ) {
                *distribution = Normal::new(point.to_f64().unwrap(), *variance).unwrap()
            }
            // Iterate over samples from this temperature
            for _ in 0..samples_per_temperature {
                // Update execution position
                execution_position_timer = update_execution_position(
                    1,
                    execution_position_timer,
                    &thread_execution_position,
                    &thread_execution_times,
                );

                // Samples new point
                let mut point = [Default::default(); N];
                for (p, r, d) in izip!(point.iter_mut(), float_ranges.iter(), distributions.iter())
                {
                    *p = sample_normal(r, d, &mut rng);
                }

                // Update execution position
                execution_position_timer = update_execution_position(
                    2,
                    execution_position_timer,
                    &thread_execution_position,
                    &thread_execution_times,
                );

                // Evaluates new point
                let value = f(&point, evaluation_data.clone());

                // Update execution position
                execution_position_timer = update_execution_position(
                    3,
                    execution_position_timer,
                    &thread_execution_position,
                    &thread_execution_times,
                );
                // Increment counter
                counter.fetch_add(1, Ordering::SeqCst);

                // Update:
                // - if there is any progression
                // - the regression `allow_change` is within a limit `rng.gen_range(0f64..1f64)`
                let difference = value - current_value;
                let allow_change = (difference / temperature).exp();
                if difference < 0. || allow_change < rng.gen_range(0f64..1f64) {
                    current_point = point;
                    current_value = value;
                    // If this value is new best value, update best value
                    if current_value < best_value {
                        best_point = current_point;
                        best_value = current_value;
                        *best.lock().unwrap() = best_value;
                    }
                }
                if thread_exit.load(Ordering::SeqCst) {
                    return (best_value, best_point);
                }
            }
            step += 1;
            temperature = cooling_schedule.decay(starting_temperature, temperature, step);
        }
        // Update execution position
        // 0 represents ended state
        thread_execution_position.store(0, Ordering::SeqCst);
        (best_value, best_point)
    }

    // Samples until value in range
    fn sample_normal<R: Rng + ?Sized, T: num::FromPrimitive>(
        range: &Range<f64>,
        distribution: &Normal<f64>,
        rng: &mut R,
    ) -> T {
        let mut point: f64 = distribution.sample(rng);
        while !range.contains(&point) {
            point = distribution.sample(rng);
        }
        T::from_f64(point).unwrap()
    }
}
