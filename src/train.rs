use std::thread;
use std::sync::Arc;
use {Dict, Matrix, Argument, Model};
use std::io::{stdout, BufReader, SeekFrom, Seek, BufRead, Write};
use std::fs::{File, metadata};
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use rand::distributions::{IndependentSample, Range};
use rand::{ThreadRng, thread_rng};
extern crate time;
use self::time::PreciseTime;
static ALL_WORDS: AtomicUsize = ATOMIC_USIZE_INIT;
static ALL_EXAMPLES: AtomicUsize = ATOMIC_USIZE_INIT;
static ALL_LOSS: AtomicUsize = ATOMIC_USIZE_INIT;

#[inline]
fn skipgram(model: &mut Model, line: &Vec<usize>, rng: &mut ThreadRng, unifrom: &Range<isize>) {
    let length = line.len() as i32;
    for w in 0..length {
        let bound = unifrom.ind_sample(rng) as i32;
        for c in -bound..bound + 1 {
            if c != 0 && w + c >= 0 && w + c < length {
                model.update(line[w as usize], line[(w + c) as usize]);
            }
        }
    }

}

fn train_thread(dict: &Dict,
                mut input: &mut Matrix,
                mut output: &mut Matrix,
                arg: Argument,
                tid: u32) {

    let between = Range::new(1, arg.win as isize);
    let mut rng = thread_rng();
    let mut model = Model::new(&mut input,
                               &mut output,
                               arg.dim,
                               arg.lr,
                               tid,
                               arg.neg,
                               dict.init_negative_table());
    let start_time = PreciseTime::now();
    let input_file = File::open(arg.input.clone()).unwrap();
    let mut reader = BufReader::with_capacity(10000, input_file);
    let file_length = metadata(arg.input).unwrap().len();
    let mut buffer = String::new();
    let mut w = Vec::new();
    let mut line = Vec::new();
    let start_bytes = (tid as f32 * (file_length as f32) / arg.nthreads as f32) as u64;
    let mut token_count = 0;
    let all_tokens = dict.ntokens * arg.epoch as usize;
    while ALL_WORDS.load(Ordering::SeqCst) < all_tokens {
        reader.seek(SeekFrom::Start(start_bytes)).expect("seek error");
        if reader.read_until(b' ', &mut w).unwrap() > 0 {
            ALL_WORDS.fetch_add(1, Ordering::SeqCst);
        }
        while reader.read_line(&mut buffer).unwrap() > 0 {
            token_count += dict.read_line(&mut buffer, &mut line);
            buffer.clear();
            // skipgram(&mut model, &line, &mut rng, &between);
            if token_count > arg.lr_update as usize {
                let words = ALL_WORDS.fetch_add(token_count, Ordering::SeqCst) as f32;
                token_count = 0;
                if tid == 0 {
                    print!("\r {}% {} {}/{}",
                           words * 100.0 / all_tokens as f32,
                           model.get_loss(),
                           words,
                           start_time.to(PreciseTime::now()));
                    stdout().flush().unwrap();
                }
            }
        }
    }
}

pub fn train(args: &Argument) {
    let dict = Arc::new(Dict::new_from_file(&args.input));
    let mut input = Matrix::new(dict.nsize(), args.dim);
    let mut output = Matrix::new(dict.nsize(), args.dim);
    input.unifrom(1.0f32 / args.dim as f32);
    output.zero();
    let input = Arc::new(input.clone());
    let output = Arc::new(output.clone());
    let mut handles = Vec::new();
    for i in 0..args.nthreads {
        let dict = dict.clone();
        let input = input.clone();
        let output = output.clone();
        let arg = args.clone();
        handles.push(thread::spawn(move || {
            let dict: &Dict = dict.as_ref();
            let input = input.as_ref().inner.get();
            let output = output.as_ref().inner.get();
            train_thread(&dict,
                         unsafe { &mut *input },
                         unsafe { &mut *output },
                         arg,
                         i);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

}