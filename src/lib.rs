use itertools::izip;
use print_duration::print_duration;
use rand::{thread_rng, Rng};
use std::{
    f64,
    io::{stdout, Write},
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

/// Random search
pub fn random_search<const N: usize>(
    iterations: usize,
    ranges: [Range<f64>; N],
    f: fn(&[f64; N]) -> f64,
    polling: Option<u64>,
) -> [f64; N] {
    // Gets cpu data
    let cpus = num_cpus::get();
    let remainder = iterations % cpus;
    let per = iterations / cpus;

    let counter = Arc::new(AtomicUsize::new(0));
    let ranges_arc = Arc::new(ranges);
    let (best_value, best_params) = search(remainder, ranges_arc.clone(), f, counter.clone());

    // (handles,counters)
    let (handles, counters): (Vec<_>, Vec<Arc<AtomicUsize>>) = (0..cpus)
        .map(|_| {
            let ranges_clone = ranges_arc.clone();
            let counter = Arc::new(AtomicUsize::new(0));
            let counter_clone = counter.clone();
            (
                thread::spawn(move || search(per, ranges_clone, f, counter_clone)),
                counter,
            )
        })
        .unzip();

    if let Some(poll_rate) = polling {
        poll(
            poll_rate,
            counters,
            counter.load(Ordering::SeqCst),
            iterations,
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

    fn search<const N: usize>(
        iterations: usize,
        ranges: Arc<[Range<f64>; N]>,
        f: fn(&[f64; N]) -> f64,
        counter: Arc<AtomicUsize>,
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
            counter.fetch_add(1, Ordering::SeqCst);
        }
        return (best_value, best_params);
    }
}

/// Grid search
pub fn grid_search<const N: usize>(
    points: [usize; N],
    ranges: [Range<f64>; N],
    f: fn(&[f64; N]) -> f64,
    polling: Option<u64>,
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

    return search_points(&point_values, f, polling);

    fn search_points<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
        polling: Option<u64>,
    ) -> [f64; N] {
        let mut start = [Default::default(); N];
        for (s, p) in start.iter_mut().zip(point_values.iter()) {
            *s = p[0];
        }
        let (_, params) = thread_search(point_values, f, start, 0, polling);
        return params;
    }
    fn thread_search<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
        mut point: [f64; N],
        index: usize,
        polling: Option<u64>,
    ) -> (f64, [f64; N]) {
        if index == point_values.len() - 1 {
            return (f(&point), point);
        }

        // (handles,counters)
        let (handles, counters): (Vec<_>, Vec<Arc<AtomicUsize>>) = point_values[index]
            .iter()
            .map(|p_value| {
                point[index] = *p_value;
                let point_values_clone = point_values.clone();
                let counter = Arc::new(AtomicUsize::new(0));
                let counter_clone = counter.clone();
                (
                    thread::spawn(move || {
                        search(&point_values_clone, f, point, index + 1, counter_clone)
                    }),
                    counter,
                )
            })
            .unzip();

        if let Some(poll_rate) = polling {
            let iterations = point_values.iter().map(|pvs| pvs.len()).product();
            poll(poll_rate, counters, 0, iterations);
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
    fn search<const N: usize>(
        point_values: &Vec<Vec<f64>>,
        f: fn(&[f64; N]) -> f64,
        mut point: [f64; N],
        index: usize,
        counter: Arc<AtomicUsize>,
    ) -> (f64, [f64; N]) {
        if index == point_values.len() {
            // panic!("hit here");
            counter.fetch_add(1, Ordering::SeqCst);
            return (f(&point), point);
        }

        let mut best_value = f64::MAX;
        let mut best_params = [Default::default(); N];
        for p_value in point_values[index].iter() {
            point[index] = *p_value;
            let (value, params) = search(point_values, f, point, index + 1, counter.clone());
            if value < best_value {
                best_value = value;
                best_params = params;
            }
        }
        return (best_value, best_params);
    }
}
fn poll(poll_rate: u64, counters: Vec<Arc<AtomicUsize>>, offset: usize, iterations: usize) {
    let start = Instant::now();
    let mut stdout = stdout();
    let mut count: usize = offset
        + counters
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .sum::<usize>();
    println!("{:20}", iterations);
    while count < iterations {
        let percent = count as f32 / iterations as f32;
        let remaining_time_estimate = start.elapsed().div_f32(percent);
        print!(
            "\r{:20} ({:.2}%) {} / {}",
            count,
            100. * percent,
            print_duration(start.elapsed(), 0..3),
            print_duration(remaining_time_estimate, 0..3)
        );
        stdout.flush().unwrap();
        thread::sleep(Duration::from_millis(poll_rate));
        count = offset
            + counters
                .iter()
                .map(|c| c.load(Ordering::SeqCst))
                .sum::<usize>();
    }

    println!(
        "\r{:20} (100.00%) {} / {}",
        count,
        print_duration(start.elapsed(), 0..3),
        print_duration(start.elapsed(), 0..3)
    );
    stdout.flush().unwrap();
}
