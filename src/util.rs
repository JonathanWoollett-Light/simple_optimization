use print_duration::print_duration;

use std::{
    io::{stdout, Write},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

pub fn poll(
    poll_rate: u64,
    counters: Vec<Arc<AtomicUsize>>,
    offset: usize,
    iterations: usize,
    early_exit_minimum: Option<f64>,
    thread_bests: Vec<Arc<Mutex<f64>>>,
    thread_exit: Arc<AtomicBool>,
) {
    let start = Instant::now();
    let mut stdout = stdout();
    let mut count: usize = offset
        + counters
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .sum::<usize>();
    println!("{:20}", iterations);

    // let mut i = 0;
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

        // In between printing, poll early exit
        if let Some(early_exit) = early_exit_minimum {
            let last_printed = Instant::now();
            while (last_printed.elapsed().as_millis() as u64) < poll_rate {
                for best in thread_bests.iter() {
                    if *best.lock().unwrap() <= early_exit {
                        thread_exit.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
        } else {
            thread::sleep(Duration::from_millis(poll_rate));
        }

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
