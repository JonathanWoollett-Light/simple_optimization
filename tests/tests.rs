#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use simple_optimization::Polling;
    // For random element we want to reruns tests a few times.
    const CHECK_ITERATIONS: usize = 100;

    fn simple_function(list: &[f64; 3], _: Option<Arc<()>>) -> f64 {
        list.iter().sum()
    }
    fn complex_function(list: &[f64; 5], _: Option<Arc<()>>) -> f64 {
        (((list[0]).powf(list[1])).sin() * list[2]) + list[3] / list[4]
    }
    fn simple_function_u8(list: &[u8; 3], _: Option<Arc<()>>) -> f64 {
        list.iter().map(|u| *u as u16).sum::<u16>() as f64
    }
    fn complex_function_u8(list: &[u8; 5], _: Option<Arc<()>>) -> f64 {
        (((list[0] as f64).powf(list[1] as f64)).sin() * list[2] as f64)
            + list[3] as f64 / list[4] as f64
    }
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
                let binary_prediction = image_pair.original_image.iter().flatten().map(|p| {
                    if *p < boundary {
                        0
                    } else {
                        255
                    }
                });
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
    #[test]
    fn random_search_simple() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                [0f64..10f64, 5f64..15f64, 10f64..20f64],
                simple_function,
                None,
                Some(Polling::new(false, Some(19.))),
                1000,
            );
            assert!(simple_function(&best, None) < 19.);
        }
    }
    #[test]
    fn random_search_simple_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                [0..10, 5..15, 10..20],
                simple_function_u8,
                None,
                Some(Polling::new(false, Some(15.))),
                1000,
            );
            assert!(simple_function_u8(&best, None) < 18.);
        }
    }
    #[test]
    fn random_search_complex() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                [
                    0f64..10f64,
                    5f64..15f64,
                    10f64..20f64,
                    15f64..25f64,
                    20f64..30f64,
                ],
                complex_function,
                None,
                Some(Polling::new(false, Some(-18.))),
                1000,
            );
            assert!(complex_function(&best, None) < -17.);
        }
    }
    #[test]
    fn random_search_complex_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                [0..10, 5..15, 10..20, 15..25, 20..30],
                complex_function_u8,
                None,
                Some(Polling::new(false, Some(-17.))),
                1000,
            );
            // -17.001623699962504
            assert!(complex_function_u8(&best, None) < -17.);
        }
    }
    #[test]
    fn random_search_boundary() {
        let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));

        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                [0..255],
                boundary_function,
                images.clone(),
                Some(Polling::new(false, Some(0.))),
                1000,
            );
            // Since we have 15 lines of 5 the error values are: 0*15, 1*15, 2*15, 3*15, 4*15, 5*15
            assert!(boundary_function(&best, images.clone()) < 1. * 15.);
        }
    }

    // Grid search
    // ---------------------------------------
    #[test]
    fn grid_search_simple() {
        let best = simple_optimization::grid_search(
            [0f64..10f64, 5f64..15f64, 10f64..20f64],
            simple_function,
            None,
            Some(Polling::new(false, Some(18.))),
            [10, 10, 10],
        );
        assert_eq!(simple_function(&best, None), 15.);
    }
    #[test]
    fn grid_search_simple_u8() {
        let best = simple_optimization::grid_search(
            [0..10, 5..15, 10..20],
            simple_function_u8,
            None,
            Some(Polling::new(false, Some(18.))),
            [10, 10, 10],
        );
        assert_eq!(simple_function_u8(&best, None), 15.);
    }
    #[test]
    fn grid_search_complex() {
        let best = simple_optimization::grid_search(
            [
                0f64..10f64,
                5f64..15f64,
                10f64..20f64,
                15f64..25f64,
                20f64..30f64,
            ],
            complex_function,
            None,
            Some(Polling::new(false, Some(-19.))),
            [4, 4, 4, 4, 4], // 4^5 = 1024 ~= 1000
        );
        assert!(complex_function(&best, None) < -14.);
    }
    #[test]
    fn grid_search_complex_u8() {
        let best = simple_optimization::grid_search(
            [0..10, 5..15, 10..20, 15..25, 20..30],
            complex_function_u8,
            None,
            Some(Polling::new(false, Some(-14.589918826094747))),
            [4, 4, 4, 4, 4], // 4^5 = 1024 ~= 1000
        );
        assert_eq!(complex_function_u8(&best, None), -14.589918826094747);
    }
    #[test]
    fn grid_search_boundary() {
        let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));

        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::grid_search(
                [0..255],
                boundary_function,
                images.clone(),
                Some(Polling::new(false, Some(0.))),
                [255],
            );
            assert_eq!(boundary_function(&best, images.clone()), 0.);
        }
    }

    // Simulated annealing
    // ---------------------------------------
    #[test]
    fn simulated_annealing_simple() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0f64..10f64, 5f64..15f64, 10f64..20f64],
                simple_function,
                None,
                Some(Polling::new(false, Some(17.))),
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                100,
                1.,
            );
            assert!(simple_function(&best, None) < 19.);
        }
    }
    #[test]
    fn simulated_annealing_simple_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0..10, 5..15, 10..20],
                simple_function_u8,
                None,
                Some(Polling::new(false, Some(16.))),
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                100,
                1.,
            );
            assert!(simple_function_u8(&best, None) < 18.);
        }
    }
    #[test]
    fn simulated_annealing_complex() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [
                    0f64..10f64,
                    5f64..15f64,
                    10f64..20f64,
                    15f64..25f64,
                    20f64..30f64,
                ],
                complex_function,
                None,
                Some(Polling::new(false, Some(-20.))),
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                100,
                1.,
            );
            assert!(complex_function(&best, None) < -17.);
        }
    }
    #[test]
    fn simulated_annealing_complex_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0..10, 5..15, 10..20, 15..25, 20..30],
                complex_function_u8,
                None,
                Some(Polling::new(false, Some(-19.))),
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                100,
                1.,
            );
            assert!(complex_function_u8(&best, None) < -17.);
        }
    }
    #[test]
    fn simulated_annealing_boundary() {
        let images: Option<Arc<Vec<ImagePair>>> = Some(Arc::new(ImagePair::new_set()));

        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0..255],
                boundary_function,
                images.clone(),
                Some(Polling::new(false, Some(0.))),
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                100,
                1.,
            );
            assert_eq!(boundary_function(&best, images.clone()), 0.);
        }
    }
}
