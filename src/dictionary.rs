use std::io;
use std::io::BufReader;
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;

const MAX_VOCAB_SIZE: usize = 30000000;

pub struct Dict<'a> {
    word2ent: HashMap<String, Entry>,
    idx2word: HashMap<usize, &'a str>,
    ntokens: usize,
    size: usize,
}
struct Entry {
    index: usize,
    count: u32,
}

pub fn hash(word: &str) -> usize {
    let mut h = 2166136261usize;
    for w in word.bytes() {
        h = h ^ (w as usize);
        h = h.wrapping_add(16777619);
    }
    h % MAX_VOCAB_SIZE
}

impl<'a> Dict<'a> {
    pub fn new() -> Dict<'a> {
        Dict {
            word2ent: HashMap::new(),
            idx2word: HashMap::new(),
            ntokens: 0,
            size: 0,
        }
    }
    #[inline]
    fn add_to_dict(words: &mut HashMap<String, Entry>, word: &str, size: &mut usize) {
        if !words.contains_key(word) {
            words.insert(word.to_string(),
                         Entry {
                             index: *size,
                             count: 1,
                         });

            *size += 1;
        } else {
            if let Some(x) = words.get_mut(word) {
                x.count += 1;
            }
        }
    }

    pub fn read_from_file(filename: &str) -> Dict {
        let mut input_file = File::open(filename).unwrap();
        let mut reader = BufReader::with_capacity(10000, input_file);
        let mut buf_str = String::with_capacity(5000);
        let mut words: HashMap<String, Entry> = HashMap::new();
        let mut idx2word: HashMap<usize, &str> = HashMap::new();
        let mut ntokens = 0;
        let mut size = 0;
        while reader.read_line(&mut buf_str).unwrap() > 0 {
            for word in buf_str.split_whitespace() {
                Dict::add_to_dict(&mut words, word, &mut size);
                if ntokens % 1000000 == 0 {
                    print!("\r read {}M words", ntokens / 1000000);
                    io::stdout().flush().ok().expect("Could not flush stdout");
                }
            }
            buf_str.clear();
        }
        size = 0;
        let mut word2ent: HashMap<String, Entry> =
            words.into_iter().filter(|&(_, ref v)| v.count >= 5).collect();
        for (_, v) in word2ent.iter_mut() {
            v.index = size;
            size += 1;
        }
        for (k, v) in &word2ent {
            idx2word.insert(v.index, &k);
        }
        println!("\r Read {} M words", ntokens / 1000000);
        Dict {
            word2ent: word2ent,
            idx2word: idx2word,
            ntokens: ntokens,
            size: size,
        }
    }
}
