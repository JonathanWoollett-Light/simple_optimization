use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::Arc;

fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 {
    // thread::sleep(Duration::from_millis(25));
    list.iter().sum()
}
fn complex_function(list: &[f64; 5], _: Option<Arc<()>>) -> f64 {
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
const SIMULATED_ANNEALING_LIMIT: (f64, f64, u32) = (100., 1., 1000); // = 100 * 10000

const SIMPLE_EXIT: f64 = 18.;
const COMPLEX_EXIT: f64 = -17.;

const BIG_LIMIT: u32 = 10000000; // 10 million
const GRID_BIG_LIMIT: u32 = 215; // BIG_LIMIT.powf(1. / 3.);

struct ImagePair {
    original_image: Vec<Vec<u8>>,
    binary_target: Vec<Vec<u8>>,
}
const IMAGE_SET: [([[u8; 5]; 5], [[u8; 5]; 5]); 3] = [
    (
        [
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
        ],
        [
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
        ],
    ),
    (
        [
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
        ],
        [
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
        ],
    ),
    (
        [
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
            [80, 120, 240, 30, 250],
        ],
        [
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
            [0, 255, 255, 0, 255],
        ],
    ),
];
impl ImagePair {
    fn new_set() -> Vec<ImagePair> {
        IMAGE_SET
            .iter()
            .map(|s| ImagePair::new(s.clone()))
            .collect()
    }
    fn new((original_image, binary_target): ([[u8; 5]; 5], [[u8; 5]; 5])) -> Self {
        Self {
            original_image: ImagePair::slice_to_vec(original_image),
            binary_target: ImagePair::slice_to_vec(binary_target),
        }
    }
    fn slice_to_vec(slice: [[u8; 5]; 5]) -> Vec<Vec<u8>> {
        slice.iter().map(|s| s.to_vec()).collect()
    }
}
fn boundary_function(list: &[u8; 1], images: Option<Arc<Vec<ImagePair>>>) -> f64 {
    let boundary = list[0];
    images
        .unwrap()
        .iter()
        .map(|image_pair| {
            let binary_prediction =
                image_pair.original_image.iter().flatten().map(
                    |p| {
                        if *p < boundary {
                            0
                        } else {
                            255
                        }
                    },
                );
            image_pair
                .binary_target
                .iter()
                .flatten()
                .zip(binary_prediction)
                .map(|(target, prediction)| (*target as i16 - prediction as i16).abs() as u64)
                .sum::<u64>()
        })
        .sum::<u64>() as f64
}

// Random search
// ---------------------------------------
fn random_search_simple_function() {
    let _best = simple_optimization::random_search(
        LIMIT,
        [0f64..10f64, 5f64..15f64, 10f64..20f64],
        simple_function,
        None,
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
        None,
        Some(COMPLEX_EXIT),
    );
}
fn random_search_boundary() {
    let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));
    let _best = simple_optimization::random_search(
        1000,
        [0..255],
        boundary_function,
        images.clone(),
        None,
        Some(0.),
    );
}
fn random_search_big() {
    let _best = simple_optimization::random_search(
        BIG_LIMIT, // billion
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
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
        None,
        Some(COMPLEX_EXIT),
    );
}
fn grid_search_boundary() {
    let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));
    let _best = simple_optimization::grid_search(
        [255],
        [0..255],
        boundary_function,
        images.clone(),
        None,
        Some(0.),
    );
}
fn grid_search_big() {
    let _best = simple_optimization::grid_search(
        [GRID_BIG_LIMIT, GRID_BIG_LIMIT, GRID_BIG_LIMIT],
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
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
        None,
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
        None,
        SIMULATED_ANNEALING_LIMIT.0,
        SIMULATED_ANNEALING_LIMIT.1,
        simple_optimization::CoolingSchedule::Fast,
        1.,
        SIMULATED_ANNEALING_LIMIT.2,
        None,
        Some(COMPLEX_EXIT),
    );
}
fn simulated_annealing_boundary() {
    let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));
    let _best = simple_optimization::simulated_annealing(
        [0..255],
        boundary_function,
        images.clone(),
        100.,
        1.,
        simple_optimization::CoolingSchedule::Fast,
        1.,
        10,
        None,
        Some(0.),
    );
}
fn simulated_annealing_big() {
    let _best = simple_optimization::simulated_annealing(
        [0f64..1f64, 0f64..1f64, 0f64..1f64],
        simple_function,
        None,
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
    c.bench_function("random_search_boundary", |b| {
        b.iter(|| random_search_boundary())
    });
    c.bench_function("random_search_big", |b| b.iter(|| random_search_big()));
    c.bench_function("grid_search_simple_function", |b| {
        b.iter(|| grid_search_simple_function())
    });
    c.bench_function("grid_search_complex_function", |b| {
        b.iter(|| grid_search_complex_function())
    });
    c.bench_function("grid_search_boundary", |b| {
        b.iter(|| grid_search_boundary())
    });
    c.bench_function("grid_search_big", |b| b.iter(|| grid_search_big()));
    c.bench_function("simulated_annealing_simple_function", |b| {
        b.iter(|| simulated_annealing_simple_function())
    });
    c.bench_function("simulated_annealing_complex_function", |b| {
        b.iter(|| simulated_annealing_complex_function())
    });
    c.bench_function("simulated_annealing_boundary", |b| {
        b.iter(|| simulated_annealing_boundary())
    });
    c.bench_function("simulated_annealing_big", |b| {
        b.iter(|| simulated_annealing_big())
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
