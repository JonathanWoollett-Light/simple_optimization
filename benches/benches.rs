use criterion::{criterion_group, criterion_main, Criterion};

fn simple_function(list: &[f64; 3]) -> f64 {
    list.iter().sum()
}
fn middle_simple_function(list: &[f64; 5]) -> f64 {
    list.iter().sum()
}
fn wider_simple_function(list: &[f64; 9]) -> f64 {
    list.iter().sum()
}
fn small_random_search() {
    let _best = simple_optimization::random_search(
        1000, // thousand
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn medium_random_search() {
    let _best = simple_optimization::random_search(
        1000000, // million
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn big_random_search() {
    let _best = simple_optimization::random_search(
        1000000000, // billion
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn small_grid_search() {
    let _best = simple_optimization::grid_search(
        [10, 10, 10], // 10^3 = 1,000 = thousand
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn medium_grid_search() {
    let _best = simple_optimization::grid_search(
        [100, 100, 100], // 100^3 = 1,000,000 = million
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn deep_big_grid_search() {
    let _best = simple_optimization::grid_search(
        [1000, 1000, 1000], // 1000^3 =  billion
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
    );
}
fn wide_big_grid_search() {
    let _best = simple_optimization::grid_search(
        [10, 10, 10, 10, 10, 10, 10, 10, 10], // 10^9 = billion
        [
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
            0f64..1f64,
        ],
        wider_simple_function,
        None,
    );
}
fn huge_grid_search() {
    let _best = simple_optimization::grid_search(
        [100, 100, 100, 100, 100], // 100^5 = 10 billion
        [0f64..1f64, 0f64..1f64, 0f64..1f64, 0f64..1f64, 0f64..1f64],
        middle_simple_function,
        None,
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("small_random_search", |b| b.iter(|| small_random_search()));
    c.bench_function("medium_random_search", |b| {
        b.iter(|| medium_random_search())
    });
    c.bench_function("big_random_search", |b| b.iter(|| big_random_search()));
    c.bench_function("small_grid_search", |b| b.iter(|| small_grid_search()));
    c.bench_function("medium_grid_search", |b| b.iter(|| medium_grid_search()));
    c.bench_function("deep_big_grid_search", |b| {
        b.iter(|| deep_big_grid_search())
    });
    c.bench_function("wide_big_grid_search", |b| {
        b.iter(|| wide_big_grid_search())
    });
    c.bench_function("huge_grid_search", |b| b.iter(|| huge_grid_search()));
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
