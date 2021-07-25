use itertools::izip;
use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};
use rand_distr::{Distribution, Normal};

use std::{
    f64,
    ops::{Range, Sub},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread,
};

use crate::util::poll;

pub enum CoolingSchedule {
    Logarithmic,
    Exponential(f64),
    Fast,
}
impl CoolingSchedule {
    fn decay(&self, t_start: f64, t_current: f64, step: u32) -> f64 {
        match self {
            Self::Logarithmic => t_current * (2f64).ln() / ((step + 1) as f64).ln(),
            Self::Exponential(x) => x * t_current,
            Self::Fast => t_start / step as f64,
        }
    }
    // Given temperature start and temperature min, gives number of steps of decay which will occur
    //  before temperature start decays to be less than temperature min, then causing program to exit.
    fn steps(&self, t_start: f64, t_min: f64) -> u32 {
        match self {
            Self::Logarithmic => (((2f64).ln() * t_start / t_min).exp() - 1f64).ceil() as u32,
            Self::Exponential(x) => ((t_min / t_start).log(*x)).ceil() as u32,
            Self::Fast => (t_start / t_min).ceil() as u32,
        }
    }
}

// TODO Multi-thread this
/// Simulated annealing
/// ```
/// fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 {
///  list.iter().sum()
/// }
/// let best = simple_optimization::simulated_annealing(
///     [0f64..10f64, 5f64..15f64, 10f64..20f64],
///     simple_function,
///     None,
///     100.,
///     1.,
///     simple_optimization::CoolingSchedule::Fast,
///     1.,
///     10,
///     None,
///     Some(17.),
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
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    starting_temperature: f64,
    minimum_temperature: f64,
    cooling_schedule: CoolingSchedule,
    variance: f64,
    samples_per_temperature: u32,
    polling: Option<u64>,
    early_exit_minimum: Option<f64>,
) -> [T; N] {
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
    let f64_ranges: Vec<Range<f64>> = ranges
        .iter()
        .map(|r| r.start.to_f64().unwrap()..r.end.to_f64().unwrap())
        .collect();

    // Variances scaled to the different ranges.
    let scaled_variances: Vec<f64> = f64_ranges
        .iter()
        .map(|r| (r.end - r.start) * variance)
        .collect();

    let steps = cooling_schedule.steps(starting_temperature, minimum_temperature);
    let iterations = steps * samples_per_temperature;
    let counter = Arc::new(AtomicU32::new(0));

    let counter_clone = counter.clone();
    let thread_best = Arc::new(Mutex::new(f64::MAX));
    let thread_exit = Arc::new(AtomicBool::new(false));

    let handle = if let Some(poll_rate) = polling {
        let thread_best_clone = thread_best.clone();
        let thread_exit_clone = thread_exit.clone();
        Some(thread::spawn(move || {
            poll(
                poll_rate,
                vec![counter_clone],
                0,
                iterations,
                early_exit_minimum,
                vec![thread_best_clone],
                thread_exit_clone,
            )
        }))
    } else {
        None
    };

    let mut step = 1;
    let mut temperature = starting_temperature;
    while temperature >= minimum_temperature {
        // Distributions to sample from at this temperature.
        let distributions: Vec<Normal<f64>> = scaled_variances
            .iter()
            .zip(current_point.iter())
            .map(|(v, p)| Normal::new(p.to_f64().unwrap(), *v).unwrap())
            .collect();
        for _ in 0..samples_per_temperature {
            // Samples new point
            let mut point = [Default::default(); N];
            for (p, r, d) in izip!(point.iter_mut(), f64_ranges.iter(), distributions.iter()) {
                *p = sample_normal(r, d, &mut rng);
            }
            let value = f(&point, evaluation_data.clone());
            counter.fetch_add(1, Ordering::SeqCst);

            let difference = value - current_value;

            let allow_change = (difference / temperature).exp();

            // Update:
            // - if there is any progression
            // - the regression `allow_change` is within a limit `rng.gen_range(0f64..1f64)`
            if difference < 0. || allow_change < rng.gen_range(0f64..1f64) {
                current_point = point;
                current_value = value;
                // If this value is new best value, update best value
                if current_value < best_value {
                    best_point = current_point;
                    best_value = current_value;
                    *thread_best.lock().unwrap() = best_value;
                }
            }
            if thread_exit.load(Ordering::SeqCst) {
                return best_point;
            }
        }
        step += 1;
        temperature = cooling_schedule.decay(starting_temperature, temperature, step);
    }

    if let Some(h) = handle {
        h.join().unwrap();
        println!();
    }
    return best_point;

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
