use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};
use std::{convert::TryInto, f64, fmt, ops::Range, sync::Arc};

use crate::util::Polling;

use friedrich::gaussian_process::GaussianProcess;
use num::ToPrimitive;
use statrs::distribution::ContinuousCDF;

// Run a test for this with `cargo test bayesian_optimization_simple -- --nocapture`
pub fn bayesian_optimization<
    A: 'static + Send + Sync,
    T: 'static
        + Copy
        + Send
        + Sync
        + Default
        + SampleUniform
        + PartialOrd
        + ToPrimitive
        + num::FromPrimitive
        + fmt::Debug,
    const N: usize,
>(
    // Generics
    ranges: [Range<T>; N],
    f: fn(&[T; N], Option<Arc<A>>) -> f64,
    evaluation_data: Option<Arc<A>>,
    _polling: Option<Polling>,
    _threads: Option<usize>,
    // Specifics
    iterations: u64,      // Number of optimization iterations
    initial_samples: u64, // Initial samples to evaluate to fit gaussian process.
    samples: u64,         // Number of samples to take for each step of fitting our gaussian process
) -> [T; N] {
    println!("1");
    let mut rng = thread_rng();
    let (mut best_value, mut best_params) = (f64::MAX, [Default::default(); N]);

    let (mut known_inputs, mut known_outputs) = (0..initial_samples)
        .map(|_| {
            let params: [T; N] = ranges
                .iter()
                .cloned()
                .map(|r| rng.gen_range(r))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let value = f(&params, evaluation_data.clone());
            if value < best_value {
                best_value = value;
                best_params = params;
            }
            (
                params
                    .iter()
                    .map(|p| p.to_f64().unwrap())
                    .collect::<Vec<_>>(),
                value,
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    println!("2");

    let mut gp = GaussianProcess::default(known_inputs.clone(), known_outputs.clone());

    println!("3");

    let normal = statrs::distribution::Normal::new(0., 1.).unwrap();

    for _ in 0..iterations {
        // Generate `sample` sample inputs
        let mut sample_inputs: Vec<Vec<f64>> = (0..samples)
            .map(|_| {
                ranges
                    .iter()
                    .cloned()
                    .map(|r| rng.gen_range(r).to_f64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Predicts the outputs mean and variance with our gaussian process for our already
        //  fully evaluated samples
        let (known_mean, _known_variance) = gp.predict_mean_variance(&known_inputs);
        let best_mean = known_mean
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        // Calculates the probability for each of our samples for it improving upon our current best.
        let (sample_mean, sample_variance) = gp.predict_mean_variance(&sample_inputs);
        let x = sample_mean
            .iter()
            .zip(sample_variance.iter())
            .map(|(mean, variance)| (mean - best_mean) / (variance + f64::EPSILON))
            .collect::<Vec<_>>();
        let probabilities = x.into_iter().map(|a| normal.cdf(a)).collect::<Vec<_>>();
        // Selects sample with highest probability of being an improvement
        let (best_index, _) = probabilities
            .into_iter()
            .enumerate()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let best_sample_vec = sample_inputs.remove(best_index);
        let best_sample: [T; N] = best_sample_vec
            .iter()
            .map(|&s| T::from_f64(s).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Evaluates this sample
        let value = f(&best_sample, evaluation_data.clone());
        if value < best_value {
            best_value = value;
            best_params = best_sample;
        }
        println!("{:.3}", best_value);
        // Not certain which is best approach, this:
        {
            // Add our newly evaluated sample to our dataset
            known_inputs.push(best_sample_vec);
            known_outputs.push(value);
            // Refit our gaussian process
            gp = GaussianProcess::default(known_inputs.clone(), known_outputs.clone());
        }
        // Or this:
        // gp.add_samples(&best_sample_vec, &value);

        // println!("4");

        // println!("5");
    }

    best_params
}
