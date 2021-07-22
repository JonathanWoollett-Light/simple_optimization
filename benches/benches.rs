use criterion::{criterion_group, criterion_main, Criterion};

fn simple_function(list: &[f64; 3]) -> f64 {
    list.iter().sum()
}
fn random_search() {
    let _best = simple_optimization::random_search(
        10000,
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        &simple_function,
    );
}
fn grid_search() {
    let _best = simple_optimization::grid_search(
        [20,20,20],
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        &simple_function,
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("random_search", |b| {
        b.iter(|| random_search())
    });
    c.bench_function("grid_search", |b| {
        b.iter(|| grid_search())
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);