use itertools::izip;
use rand::{thread_rng, Rng};
use std::thread;
use std::{f64, ops::Range, sync::Arc};

/// Random search
pub fn random_search<const N: usize>(
    iterations: usize,
    ranges: [Range<f64>; N],
    f: fn(&[f64; N]) -> f64,
) -> [f64; N] {
    // Gets cpu data
    let cpus = num_cpus::get();
    let remainder = iterations % cpus;
    let per = iterations / cpus;

    let ranges_arc = Arc::new(ranges);
    let (best_value, best_params) = search(remainder, ranges_arc.clone(), f);

    let handles: Vec<_> = (0..cpus)
        .map(|_| {
            let ranges_clone = ranges_arc.clone();
            thread::spawn(move || search(per, ranges_clone, f))
        })
        .collect();
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

    fn search<const N: usize>(
        iterations: usize,
        ranges: Arc<[Range<f64>; N]>,
        f: fn(&[f64; N]) -> f64,
    ) -> (f64, [f64; N]) {
        let mut rng = thread_rng();
        let mut params = [Default::default(); N];

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for _ in 0..iterations {
            // Gen random values
            for (range, param) in ranges.iter().zip(params.iter_mut()) {
                *param = rng.gen_range(range.clone())
            }
            // Run function
            let new_value = f(&params);
            // Check best
            if new_value < best_value {
                best_value = new_value;
                best_params = params;
            }
        }
        return (best_value, best_params);
    }
}

/// Grid search
pub fn grid_search<const N: usize>(
    points: [usize; N],
    ranges: [Range<f64>; N],
    f: fn(&[f64; N]) -> f64,
) -> [f64; N] {
    // Compute step sizes
    let mut steps = [Default::default(); N];
    for (r, k, s) in izip!(ranges.iter(), points.iter(), steps.iter_mut()) {
        *s = (r.end - r.start) / *k as f64;
    }

    // Compute point values
    let point_values: Vec<Vec<f64>> = izip!(ranges.iter(), points.iter(), steps.iter())
        .map(|(r, k, s)| {
            (0..*k)
                .scan(r.start, |state, _| {
                    *state += *s;
                    Some(*state)
                })
                .collect()
        })
        .collect();

    return search_points(&point_values, f);

    fn search_points<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
    ) -> [f64; N] {
        let mut start = [Default::default(); N];
        for (s, p) in start.iter_mut().zip(point_values.iter()) {
            *s = p[0];
        }
        let (_, params) = thread_search(point_values, f, start, 0);
        return params;
    }
    fn thread_search<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
        mut point: [f64; N],
        index: usize,
    ) -> (f64, [f64; N]) {
        if index == point_values.len() - 1 {
            return (f(&point), point);
        }

        let handles: Vec<_> = point_values[index]
            .iter()
            .map(|p_value| {
                point[index] = *p_value;
                let point_values_clone = point_values.clone();
                thread::spawn(move || search(&point_values_clone, f, point, index + 1))
            })
            .collect();
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
    fn search<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
        mut point: [f64; N],
        index: usize,
    ) -> (f64, [f64; N]) {
        if index == point_values.len() - 1 {
            return (f(&point), point);
        }

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for p_value in point_values[index].iter() {
            point[index] = *p_value;
            let (value, params) = search(point_values, f, point, index + 1);
            if value < best_value {
                best_value = value;
                best_params = params;
            }
        }
        return (best_value, best_params);
    }
}
