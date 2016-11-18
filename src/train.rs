use std::thread;
use std::sync::Arc;
use {Dict, Matrix, Argument, Model};
use std::io::{BufReader, SeekFrom, Seek, Read, BufRead};
use std::fs::{File, metadata};
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use rand::distributions::{IndependentSample, Range};
use rand::{ThreadRng, thread_rng};

static ALL_WORDS: AtomicUsize = ATOMIC_USIZE_INIT;
static ALL_EXAMPLES: AtomicUsize = ATOMIC_USIZE_INIT;
static ALL_LOSS: AtomicUsize = ATOMIC_USIZE_INIT;

#[inline]
fn skipgram(model: &mut Model,
            line: &Vec<usize>,
            loss: &mut f32,
            nsamples: &mut u32,
            rng: &mut ThreadRng,
            unifrom: &Range<isize>) {
    let length = line.len() as i32;
    for w in 0..length {
        let bound = unifrom.ind_sample(rng) as i32;
        for c in -bound..bound + 1 {
            if c != 0 && w + c >= 0 && w + c < length {
                *loss += model.update(line[w as usize], line[(w + c) as usize]);
                *nsamples += 1;
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

    let input_file = File::open(arg.input.clone()).unwrap();
    let mut reader = BufReader::with_capacity(10000, input_file);
    let file_length = metadata(arg.input).unwrap().len();
    let mut buffer = String::new();
    let mut w = Vec::new();
    let mut loss = 0f32;
    let mut nsamples = 1;
    let mut line = Vec::new();
    let start_bytes = tid as u64 * file_length / arg.nthreads as u64;
    let mut token_count = 0;
    reader.seek(SeekFrom::Start(start_bytes));
    if reader.read_until(b' ', &mut w).unwrap() > 0 {
        ALL_WORDS.fetch_add(1, Ordering::SeqCst);
    }
    loop {
        let line_count = reader.read_line(&mut buffer).unwrap();
        token_count += line_count;
        dict.read_line(&mut buffer, &mut line);

        skipgram(&mut model,
                 &line,
                 &mut loss,
                 &mut nsamples,
                 &mut rng,
                 &between);
        // TODO : delete as usize
        if (token_count > arg.lr_update as usize) {
            ALL_WORDS.fetch_add(token_count, Ordering::SeqCst);
            print!("\r{}", loss);
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
            println!("{}:{}", i, dict.nsize());
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