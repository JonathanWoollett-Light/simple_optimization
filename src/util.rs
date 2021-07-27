use print_duration::print_duration;

use std::{
    io::{stdout, Write},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

pub struct Polling {
    pub poll_rate: u64,
    pub printing: bool,
    pub early_exit_minimum: Option<f64>
}
impl Polling {
    const DEFAULT_POLL_RATE: u64 = 10;
    pub fn new(printing: bool, early_exit_minimum: Option<f64>) -> Self {
        Self {
            poll_rate: Polling::DEFAULT_POLL_RATE,
            printing,
            early_exit_minimum
        }
    }
}

pub fn poll(
    data: Polling,
    counters: Vec<Arc<AtomicU32>>,
    offset: u32,
    iterations: u32,
    thread_bests: Vec<Arc<Mutex<f64>>>,
    thread_exit: Arc<AtomicBool>,
) {
    let start = Instant::now();
    let mut stdout = stdout();
    let mut count = offset
        + counters
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .sum::<u32>();
            
    if data.printing {
        println!("{:20}", iterations);
    }


    let mut poll_time = Instant::now();
    let mut held_best: f64 = f64::MAX;
    while count < iterations {

        if data.printing {
            // loop {
            let percent = count as f32 / iterations as f32;

            // If count == 0, give 00... for remaining time as placeholder
            let remaining_time_estimate = if count == 0 {
                Duration::new(0, 0)
            } else {
                start.elapsed().div_f32(percent)
            };
            print!(
                "\r{:20} ({:.2}%) {} / {} [{}]",
                count,
                100. * percent,
                print_duration(start.elapsed(), 0..3),
                print_duration(remaining_time_estimate, 0..3),
                if held_best == f64::MAX { String::from("?") } else { format!("{}",held_best) }
            );
            stdout.flush().unwrap();
        }


        // Updates best and does early exiting
        match (data.early_exit_minimum, data.printing) {
            (Some(early_exit),true) => {
                for thread_best in thread_bests.iter() {
                    let thread_best_temp = *thread_best.lock().unwrap();
                    if thread_best_temp < held_best {
                        held_best = thread_best_temp;
                        if thread_best_temp <= early_exit {
                            thread_exit.store(true, Ordering::SeqCst);
                            println!();
                            return;
                        }
                    }
                    
                }
            },
            (None, true) => {
                for thread_best in thread_bests.iter() {
                    let thread_best_temp = *thread_best.lock().unwrap();
                    if thread_best_temp < held_best {
                        held_best = thread_best_temp;
                    }
                }
            },
            (Some(early_exit), false) => {
                for thread_best in thread_bests.iter() {
                    if *thread_best.lock().unwrap() <= early_exit {
                        thread_exit.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            },
            (None,false) => {}
        }
            
        thread::sleep(saturating_sub(
            Duration::from_millis(data.poll_rate),
            poll_time.elapsed(),
        ));
        poll_time = Instant::now();

        count = offset
            + counters
                .iter()
                .map(|c| c.load(Ordering::SeqCst))
                .sum::<u32>();
    }

    if data.printing {
        println!(
            "\r{:20} (100.00%) {} / {} [{}]",
            count,
            print_duration(start.elapsed(), 0..3),
            print_duration(start.elapsed(), 0..3),
            held_best
        );
        stdout.flush().unwrap();
    }
}
// Since `Duration::saturating_sub` is unstable this is an alternative.
fn saturating_sub(a: Duration, b: Duration) -> Duration {
    if let Some(dur) = a.checked_sub(b) {
        dur
    } else {
        Duration::new(0, 0)
    }
}
