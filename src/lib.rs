use itertools::izip;
use rand::{thread_rng, Rng};
use std::{f64, ops::Range};

/// Random search
pub fn random_search<const N: usize>(
    iterations: usize,
    ranges: [Range<f64>; N],
    f: &dyn Fn(&[f64; N]) -> f64,
) -> [f64; N] {
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

    return best_params;
}

/// Grid search
pub fn grid_search<const N: usize>(
    points: [usize; N],
    ranges: [Range<f64>; N],
    f: &dyn Fn(&[f64; N]) -> f64,
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

    return search_points(&point_values,f);

    fn search_points<const N:usize>(point_values: &Vec<Vec<f64>>, f: &dyn Fn(&[f64; N]) -> f64) -> [f64;N] {
        let mut start = [Default::default();N];
        for (s,p) in start.iter_mut().zip(point_values.iter()) {
            *s = p[0];
        }
        search(point_values, f, start, 0)
    }
    fn search<const N:usize>(point_values: &Vec<Vec<f64>>, f: &dyn Fn(&[f64; N]) -> f64, mut point: [f64;N], index: usize) -> [f64;N] {
        if index == point_values.len() - 1 { return point.clone(); }

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default();N];
        for p_value in point_values[index].iter() {
            point[index] = *p_value;
            let params = search(point_values, f, point, index + 1);
            let value = f(&params);
            if value < best_value {
                best_value = value;
                best_params = params;
            }
        }
        return best_params;
    }
}
