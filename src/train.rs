use std::thread;
use std::sync::Arc;
use {Dict, Matrix, Argument, Model};
use std::io::{BufReader, SeekFrom, Seek, BufRead};
use std::fs::{File, metadata};
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use rand::distributions::{IndependentSample, Range};
use rand::StdRng;
use time::PreciseTime;
use Word2vec;
use W2vError;
static ALL_WORDS: AtomicUsize = ATOMIC_USIZE_INIT;

fn skipgram(model: &mut Model, line: &Vec<usize>, rng: &mut StdRng, unifrom: &Range<isize>) {
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
                tid: u32,
                neg_table: Arc<Vec<usize>>)
                -> Result<bool, W2vError> {

    let between = Range::new(1, (arg.win + 1) as isize);
    let mut rng = StdRng::new().unwrap();
    let mut model = Model::new(&mut input, &mut output, arg.dim, arg.lr, arg.neg, neg_table);
    let start_time = PreciseTime::now();
    let input_file = try!(File::open(arg.input.clone()));
    let mut reader = BufReader::with_capacity(10000, input_file);
    let file_length = metadata(arg.input)?.len();
    let mut buffer = String::new();
    let mut w = Vec::new();
    let mut line: Vec<usize> = Vec::new();
    let start_bytes = (tid as f32 * (file_length as f32) / arg.nthreads as f32) as u64;
    let (mut token_count, mut local_all_count, mut epoch) = (0, 2, 0);
    let all_tokens = dict.ntokens * arg.epoch as usize;
    let thread_token = (dict.ntokens as f32 / arg.nthreads as f32) as usize;
    while epoch < arg.epoch {
        try!(reader.seek(SeekFrom::Start(start_bytes)));
        if reader.read_until(b' ', &mut w).unwrap() > 0 {
            ALL_WORDS.fetch_add(1, Ordering::SeqCst);
            local_all_count = 2;
            w.clear();
        }
        loop {
            if local_all_count >= thread_token {
                epoch += 1;
                break;
            }
            reader.read_line(&mut buffer).unwrap();
            let token = dict.read_line(&mut buffer, &mut line);
            local_all_count += token;
            token_count += token;
            buffer.clear();
            skipgram(&mut model, &line, &mut rng, &between);
            line.clear();
            if token_count > arg.lr_update as usize {
                let words = ALL_WORDS.fetch_add(token_count, Ordering::SeqCst) as f32;
                let progress = words / all_tokens as f32;
                model.set_lr(arg.lr * (1.0 - progress));
                token_count = 0;
                if tid == 0 {
                    if arg.verbose {
                        info!("\rProgress:{:.1}% words/sec/thread:{:<7.0} lr:{:.4} loss:{:.5}",
                           progress * 100.,
                           ((words * 1000.) /
                            (start_time.to(PreciseTime::now())
                               .num_milliseconds() as f32)) as u64,
                           model.get_lr(),
                           model.get_loss());
                    }
                }
            }
        }
    }
    ALL_WORDS.fetch_add(token_count, Ordering::SeqCst);
    Ok(true)
}

pub fn train(args: &Argument) -> Result<Word2vec, W2vError> {
    let dict = try!(Dict::new_from_file(&args.input, args.min_count, args.threshold));

    let dict = Arc::new(dict);
    let mut input_mat = Matrix::new(dict.nsize(), args.dim);
    let mut output_mat = Matrix::new(dict.nsize(), args.dim);

    input_mat.unifrom(1.0f32 / args.dim as f32);
    output_mat.zero();
    let input = Arc::new(input_mat.make_send());
    let output = Arc::new(output_mat.make_send());
    let neg_table = dict.init_negative_table();
    let mut handles = Vec::new();
    for i in 0..args.nthreads {
        let (input, output, dict, arg, neg_table) =
            (input.clone(), output.clone(), dict.clone(), args.clone(), neg_table.clone());

        handles.push(thread::spawn(move || {
            let dict: &Dict = dict.as_ref();
            let input = input.as_ref().inner.get();
            let output = output.as_ref().inner.get();
            train_thread(&dict,
                         unsafe { &mut *input },
                         unsafe { &mut *output },
                         arg,
                         i,
                         neg_table)
        }));
    }
    for h in handles {
        try!(h.join().unwrap());
    }

    let input = Arc::try_unwrap(input).unwrap();
    let output = Arc::try_unwrap(output).unwrap();
    let dict = Arc::try_unwrap(dict).unwrap();
    let w2v = Word2vec::new(unsafe { input.inner.into_inner() },
                            unsafe { output.inner.into_inner() },
                            args.dim,
                            dict);
    Ok(w2v)
}