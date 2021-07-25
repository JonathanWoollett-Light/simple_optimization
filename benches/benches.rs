use criterion::{criterion_group, criterion_main, Criterion};

fn simple_function(list: &[f64; 3]) -> f64 {
    // thread::sleep(Duration::from_millis(25));
    list.iter().sum()
}
fn complex_function(list: &[f64; 5]) -> f64 {
    // thread::sleep(Duration::from_millis(25));
    (((list[0]).powf(list[1])).sin() * list[2]) + list[3] / list[4]
}
// `thread::sleep` except it keeps thread busy
//  (`thread::sleep` break criterion for some reason)
// fn thread_wait(duration: std::time::Duration) {
//     let start = std::time::Instant::now();
//     while start.elapsed() < duration {}
// }

// Timeout iterations
const LIMIT: u32 = 1000000; // 1 million
const GRID_SIMPLE_LIMIT: u32 = 100; // LIMIT.powf(1. / 3.);
const GRID_COMPLEX_LIMIT: u32 = 15; // LIMIT.powf(1. / 5.);
const SIMULATED_ANNEALING_LIMIT: (f64,f64,u32) = (100.,1.,1000); // = 100 * 10000

const SIMPLE_EXIT: f64 = 18.;
const COMPLEX_EXIT: f64 = -17.;

const BIG_LIMIT: u32 = 10000000; // 10 million
const GRID_BIG_LIMIT: u32 = 215; // BIG_LIMIT.powf(1. / 3.);

// Random search
// ---------------------------------------
fn random_search_simple_function() {
    let _best = simple_optimization::random_search(
        LIMIT,
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        simple_function,
        None,
        Some(SIMPLE_EXIT),
    );
}
fn random_search_complex_function() {
    let _best = simple_optimization::random_search(
        LIMIT,
        [
            0f64..10f64,
            5f64..15f64,
            10f64..20f64,
            25f64..35f64,
            30f64..40f64,
        ],
        complex_function,
        None,
        Some(COMPLEX_EXIT),
    );
}
fn random_search_big() {
    let _best = simple_optimization::random_search(
        BIG_LIMIT, // billion
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
        None,
    );
}

// Grid search
// ---------------------------------------
fn grid_search_simple_function() {
    let _best = simple_optimization::grid_search(
        [GRID_SIMPLE_LIMIT, GRID_SIMPLE_LIMIT, GRID_SIMPLE_LIMIT],
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        simple_function,
        None,
        Some(SIMPLE_EXIT),
    );
}
fn grid_search_complex_function() {
    let _best = simple_optimization::grid_search(
        [
            GRID_COMPLEX_LIMIT,
            GRID_COMPLEX_LIMIT,
            GRID_COMPLEX_LIMIT,
            GRID_COMPLEX_LIMIT,
            GRID_COMPLEX_LIMIT,
        ],
        [
            0f64..10f64,
            5f64..15f64,
            10f64..20f64,
            25f64..35f64,
            30f64..40f64,
        ],
        complex_function,
        None,
        Some(COMPLEX_EXIT),
    );
}
fn grid_search_big() {
    let _best = simple_optimization::grid_search(
        [GRID_BIG_LIMIT, GRID_BIG_LIMIT, GRID_BIG_LIMIT],
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
        None,
    );
}

// Simulated annealing
// ---------------------------------------
fn simulated_annealing_simple_function() {
    let _best = simple_optimization::simulated_annealing(
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        simple_function,
        SIMULATED_ANNEALING_LIMIT.0,
        SIMULATED_ANNEALING_LIMIT.1,
        simple_optimization::CoolingSchedule::Fast,
        1.,
        SIMULATED_ANNEALING_LIMIT.2,
        None,
        Some(SIMPLE_EXIT),
    );
}
fn simulated_annealing_complex_function() {
    let _best = simple_optimization::simulated_annealing(
        [
            0f64..10f64,
            5f64..15f64,
            10f64..20f64,
            25f64..35f64,
            30f64..40f64,
        ],
        complex_function,
        SIMULATED_ANNEALING_LIMIT.0,
        SIMULATED_ANNEALING_LIMIT.1,
        simple_optimization::CoolingSchedule::Fast,
        1.,
        SIMULATED_ANNEALING_LIMIT.2,
        None,
        Some(COMPLEX_EXIT),
    );
}
fn simulated_annealing_big() {
    let _best = simple_optimization::simulated_annealing(
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        100.,
        1.,
        simple_optimization::CoolingSchedule::Fast,
        1.,
        100000,
        None,
        None,
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("random_search_simple_function", |b| {
        b.iter(|| random_search_simple_function())
    });
    c.bench_function("random_search_complex_function", |b| {
        b.iter(|| random_search_complex_function())
    });
    c.bench_function("random_search_big", |b| b.iter(|| random_search_big()));
    c.bench_function("grid_search_simple_function", |b| {
        b.iter(|| grid_search_simple_function())
    });
    c.bench_function("grid_search_complex_function", |b| {
        b.iter(|| grid_search_complex_function())
    });
    c.bench_function("grid_search_big", |b| b.iter(|| grid_search_big()));
    c.bench_function("simulated_annealing_simple_function", |b| {
        b.iter(|| simulated_annealing_simple_function())
    });
    c.bench_function("simulated_annealing_complex_function", |b| {
        b.iter(|| simulated_annealing_complex_function())
    });
    c.bench_function("simulated_annealing_big", |b| {
        b.iter(|| simulated_annealing_big())
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
