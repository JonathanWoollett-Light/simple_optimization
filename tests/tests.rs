#[cfg(test)]
mod tests {
    // For random element we want to reruns tests a few times.
    const CHECK_ITERATIONS: usize = 100;

    fn simple_function_u8(list: &[u8; 3]) -> f64 {
        list.iter().map(|u| *u as u16).sum::<u16>() as f64
    }
    fn simple_function(list: &[f64; 3]) -> f64 {
        list.iter().sum()
    }
    #[rustfmt::skip]
    fn moderate_function(list: &[f64; 5]) -> f64 {
        125. * list[0].cos() - 
        list[1].powf(2.5) + 
        2. * list[2] +
        0.125 * list[3].powf(-3.5) -
        13.2 * list[4].powf(3.12)
    }

    #[test]
    fn random_search_simple() {
        const ALLOWANCE: f64 = 2.5;
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                2000,
                [0f64..10f64, 5f64..15f64, 10f64..20f64],
                simple_function,
                None,
            );
            assert!(best[0] < ALLOWANCE);
            assert!(best[1] < 5. + ALLOWANCE);
            assert!(best[2] < 10. + ALLOWANCE);
        }
    }
    #[test]
    fn random_search_moderate() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                2000,
                [
                    0f64..10f64,
                    5f64..15f64,
                    10f64..20f64,
                    15f64..25f64,
                    20f64..30f64,
                ],
                moderate_function,
                None,
            );
            println!("best: ({}) {:.?}", moderate_function(&best), best);
            assert!(moderate_function(&best) < -530000.);
        }
    }
    #[test]
    fn random_search_simple_u8() {
        for _ in 0..CHECK_ITERATIONS {
            let best = simple_optimization::random_search(
                10000,
                [0..10, 5..15, 10..20],
                simple_function_u8,
                None,
            );
            assert_eq!(best[0], 0);
            assert_eq!(best[1], 5);
            assert_eq!(best[2], 10);
        }
    }

    #[test]
    fn grid_search() {
        let best = simple_optimization::grid_search(
            [10, 10, 10],
            [0f64..10f64, 5f64..15f64, 10f64..20f64],
            simple_function,
            None,
        );
        assert!(best == [1., 6., 11.]);
    }
    #[test]
    fn grid_search_u8() {
        let best = simple_optimization::grid_search(
            [10, 10, 10],
            [0..10, 5..15, 10..20],
            simple_function_u8,
            None,
        );
        assert!(best == [1, 6, 11]);
    }
}
