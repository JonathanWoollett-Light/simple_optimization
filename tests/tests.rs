#[cfg(test)]
mod tests {
    // For random element we want to reruns tests a few times.
    const CHECK_ITERATIONS: usize = 100;

    fn simple_function(list: &[f64; 3]) -> f64 {
        list.iter().sum()
    }
    fn complex_function(list: &[f64; 5]) -> f64 {
        (((list[0]).powf(list[1])).sin() * list[2]) + list[3] / list[4]
    }
    fn simple_function_u8(list: &[u8; 3]) -> f64 {
        list.iter().map(|u| *u as u16).sum::<u16>() as f64
    }
    fn complex_function_u8(list: &[u8; 5]) -> f64 {
        (((list[0] as f64).powf(list[1] as f64)).sin() * list[2] as f64)
            + list[3] as f64 / list[4] as f64
    }

    // Random search
    // ---------------------------------------
    #[test]
    fn random_search_simple() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                1000,
                [0f64..10f64, 5f64..15f64, 10f64..20f64],
                simple_function,
                None,
                Some(19.),
            );
            assert!(simple_function(&best) < 19.);
        }
    }
    #[test]
    fn random_search_simple_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                1000,
                [0..10, 5..15, 10..20],
                simple_function_u8,
                None,
                Some(15.),
            );
            assert!(simple_function_u8(&best) < 18.);
        }
    }
    #[test]
    fn random_search_complex() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                1000,
                [
                    0f64..10f64,
                    5f64..15f64,
                    10f64..20f64,
                    15f64..25f64,
                    20f64..30f64,
                ],
                complex_function,
                None,
                Some(-18.),
            );
            assert!(complex_function(&best) < -18.);
        }
    }
    #[test]
    fn random_search_complex_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                1000,
                [0..10, 5..15, 10..20, 15..25, 20..30],
                complex_function_u8,
                None,
                Some(-17.),
            );
            // -17.001623699962504
            assert!(complex_function_u8(&best) < -17.);
        }
    }

    // Grid search
    // ---------------------------------------
    #[test]
    fn grid_search_simple() {
        let best = simple_optimization::grid_search(
            [10, 10, 10],
            [0f64..10f64, 5f64..15f64, 10f64..20f64],
            simple_function,
            None,
            Some(18.),
        );
        assert_eq!(simple_function(&best), 18.);
    }
    #[test]
    fn grid_search_simple_u8() {
        let best = simple_optimization::grid_search(
            [10, 10, 10],
            [0..10, 5..15, 10..20],
            simple_function_u8,
            None,
            Some(18.),
        );
        assert_eq!(simple_function_u8(&best), 18.);
    }
    #[test]
    fn grid_search_complex() {
        let best = simple_optimization::grid_search(
            [4, 4, 4, 4, 4], // 4^5 = 1024 ~= 1000
            [
                0f64..10f64,
                5f64..15f64,
                10f64..20f64,
                15f64..25f64,
                20f64..30f64,
            ],
            complex_function,
            None,
            Some(-19.),
        );
        assert!(complex_function(&best) < -19.);
    }
    #[test]
    fn grid_search_complex_u8() {
        let best = simple_optimization::grid_search(
            [4, 4, 4, 4, 4], // 4^5 = 1024 ~= 1000
            [0..10, 5..15, 10..20, 15..25, 20..30],
            complex_function_u8,
            None,
            Some(-17.001623699962504),
        );
        assert_eq!(complex_function_u8(&best), -17.001623699962504);
    }

    // Simulated annealing
    // ---------------------------------------
    #[test]
    fn simulated_annealing_simple() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0f64..10f64, 5f64..15f64, 10f64..20f64],
                simple_function,
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                1.,
                10,
                None,
                Some(17.),
            );
            assert!(simple_function(&best) < 19.);
        }
    }
    #[test]
    fn simulated_annealing_simple_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0..10, 5..15, 10..20],
                simple_function_u8,
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                1.,
                10,
                None,
                Some(16.),
            );
            assert!(simple_function_u8(&best) < 18.);
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
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                1.,
                10,
                None,
                Some(-20.),
            );
            assert!(complex_function(&best) < -17.);
        }
    }
    #[test]
    fn simulated_annealing_complex_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::simulated_annealing(
                [0..10, 5..15, 10..20, 15..25, 20..30],
                complex_function_u8,
                100.,
                1.,
                simple_optimization::CoolingSchedule::Fast,
                1.,
                10,
                None,
                Some(-19.),
            );
            assert!(complex_function_u8(&best) < -17.);
        }
    }
}
