use print_duration::print_duration;

use std::{
    io::{stdout, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
    convert::TryInto
};

pub fn update_execution_position<const N: usize>(
    i: usize,
    execution_position_timer: Instant,
    thread_execution_position: &Arc<AtomicU8>,
    thread_execution_times: &Arc<[Mutex<(Duration, u64)>;N]>
) -> Instant {
    {
        let mut data = thread_execution_times[i].lock().unwrap();
        data.0 += execution_position_timer.elapsed();
        data.1 += 1;
    }
    thread_execution_position.store(i as u8,Ordering::SeqCst);
    Instant::now()
}

pub struct Polling {
    pub poll_rate: u64,
    pub printing: bool,
    pub early_exit_minimum: Option<f64>,
    pub thread_execution_reporting: bool
}
impl Polling {
    const DEFAULT_POLL_RATE: u64 = 10;
    pub fn new(printing: bool, early_exit_minimum: Option<f64>) -> Self {
        Self {
            poll_rate: Polling::DEFAULT_POLL_RATE,
            printing,
            early_exit_minimum,
            thread_execution_reporting: false
        }
    }
}

pub fn poll<const N: usize>(
    data: Polling,
    // Current count of each thread.
    counters: Vec<Arc<AtomicU64>>,
    offset: u64,
    // Final total iterations.
    iterations: u64,
    // Best values of each thread.
    thread_bests: Vec<Arc<Mutex<f64>>>,
    // Early exit switch.
    thread_exit: Arc<AtomicBool>,
    // Current positions of execution of each thread.
    thread_execution_positions: Vec<Arc<AtomicU8>>,
    // Current average times between execution positions for each thread
    thread_execution_times: Vec<Arc<[Mutex<(Duration,u64)>; N]>>
) {
    let start = Instant::now();
    let mut stdout = stdout();
    let mut count = offset
        + counters
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .sum::<u64>();

    if data.printing {
        println!("{:20}", iterations);
    }

    let mut poll_time = Instant::now();
    let mut held_best: f64 = f64::MAX;

    let mut held_average_execution_times: [(Duration, u64);N] = vec![(Duration::new(0,0),0); N].try_into().unwrap();
    let mut held_recent_execution_times: [Duration;N] = vec![Duration::new(0,0); N].try_into().unwrap();
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
                "\r{:20} ({:.2}%) {} / {} [{}] {}\t",
                count,
                100. * percent,
                print_duration(start.elapsed(), 0..3),
                print_duration(remaining_time_estimate, 0..3),
                if held_best == f64::MAX {
                    String::from("?")
                } else {
                    format!("{}", held_best)
                },
                if data.thread_execution_reporting {
                    let (average_execution_times, recent_execution_times): (Vec<String>,Vec<String>) = (0..thread_execution_times[0].len()).map(|i| {
                        let (mut sum, mut num) = (Duration::new(0,0),0);
                        for n in 0..thread_execution_times.len() {
                            {
                                let mut data = thread_execution_times[n][i].lock().unwrap();
                                sum += data.0;
                                held_average_execution_times[i].0 += data.0;
                                num += data.1;
                                held_average_execution_times[i].1 += data.1;
                                *data = (Duration::new(0,0),0);
                            }
                        }
                        if num > 0 {
                            held_recent_execution_times[i] = sum.div_f64(num as f64);
                        }
                        (
                            if held_average_execution_times[i].1 > 0 {
                                format!("{:.1?}",held_average_execution_times[i].0.div_f64(held_average_execution_times[i].1 as f64))
                            }
                            else {
                                String::from("?")
                            },
                            if held_recent_execution_times[i] > Duration::new(0,0) {
                                format!("{:.1?}",held_recent_execution_times[i])
                            }
                            else {
                                String::from("?")
                            }
                        )
                    }).unzip();

                    let execution_positions: Vec<u8> = thread_execution_positions.iter().map(|pos|pos.load(Ordering::SeqCst)).collect();
                    format!("{{ [{}] [{}] {:.?} }}", recent_execution_times.join(", "), average_execution_times.join(", "), execution_positions)
                }
                else {
                    String::from("")
                }
            );
            stdout.flush().unwrap();
        }

        // Updates best and does early exiting
        match (data.early_exit_minimum, data.printing) {
            (Some(early_exit), true) => {
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
            }
            (None, true) => {
                for thread_best in thread_bests.iter() {
                    let thread_best_temp = *thread_best.lock().unwrap();
                    if thread_best_temp < held_best {
                        held_best = thread_best_temp;
                    }
                }
            }
            (Some(early_exit), false) => {
                for thread_best in thread_bests.iter() {
                    if *thread_best.lock().unwrap() <= early_exit {
                        thread_exit.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
            (None, false) => {}
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
                .sum::<u64>();
    }

    if data.printing {
        println!(
            "\r{:20} (100.00%) {} / {} [{}] {}\t",
            count,
            print_duration(start.elapsed(), 0..3),
            print_duration(start.elapsed(), 0..3),
            held_best,
            if data.thread_execution_reporting {
                let (average_execution_times, recent_execution_times): (Vec<String>,Vec<String>) = (0..thread_execution_times[0].len()).map(|i| {
                    let (mut sum, mut num) = (Duration::new(0,0),0);
                    for n in 0..thread_execution_times.len() {
                        {
                            let mut data = thread_execution_times[n][i].lock().unwrap();
                            sum += data.0;
                            held_average_execution_times[i].0 += data.0;
                            num += data.1;
                            held_average_execution_times[i].1 += data.1;
                            *data = (Duration::new(0,0),0);
                        }
                    }
                    if num > 0 {
                        held_recent_execution_times[i] = sum.div_f64(num as f64);
                    }
                    (
                        if held_average_execution_times[i].1 > 0 {
                            format!("{:.1?}",held_average_execution_times[i].0.div_f64(held_average_execution_times[i].1 as f64))
                        }
                        else {
                            String::from("?")
                        },
                        if held_recent_execution_times[i] > Duration::new(0,0) {
                            format!("{:.1?}",held_recent_execution_times[i])
                        }
                        else {
                            String::from("?")
                        }
                    )
                }).unzip();

                let execution_positions: Vec<u8> = thread_execution_positions.iter().map(|pos|pos.load(Ordering::SeqCst)).collect();
                format!("{{ [{}] [{}] {:.?} }}", recent_execution_times.join(", "), average_execution_times.join(", "), execution_positions)
            }
            else {
                String::from("")
            }
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