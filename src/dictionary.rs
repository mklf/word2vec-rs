use std::io;
use std::io::BufReader;
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;

const MAX_VOCAB_SIZE: usize = 30000000;

pub struct Dict {
    word2idx: HashMap<String, Entry>,
    idx2count: Vec<u32>,
    ntokens: usize,
    size: usize,
}
struct Entry(usize, u32);

pub fn hash(word: &str) -> usize {
    let mut h = 2166136261usize;
    for w in word.bytes() {
        h = h ^ (w as usize);
        h = h.wrapping_add(16777619);
    }
    h % MAX_VOCAB_SIZE
}

impl Dict {
    pub fn new() -> Dict {
        Dict {
            word2idx: HashMap::with_capacity(100000),
            idx2count: Vec::new(),
            ntokens: 0,
            size: 0,
        }
    }


    pub fn read_from_file(&mut self, filename: &str) {
        let mut input_file = File::open(filename).unwrap();
        let mut reader = BufReader::with_capacity(10000, input_file);
        let mut buf_str = String::with_capacity(5000);
        let mut size = self.size;
        let mut words = HashMap::new();
        while reader.read_line(&mut buf_str).unwrap() > 0 {
            for word in buf_str.split_whitespace() {
                self.ntokens += 1;
                if !words.contains_key(word) {
                    words.insert(word.to_string(), Entry(size, 1));
                    size += 1;
                } else {
                    if let Some(x) = words.get_mut(word) {
                        x.1 += 1;
                    }
                }
                if self.ntokens % 1000000 == 0 {
                    print!("\r read {}M words", self.ntokens / 1000000);
                    io::stdout().flush().ok().expect("Could not flush stdout");
                }
            }
            buf_str.clear();
        }
        let mut s = String::new();
        io::stdin().read_to_string(&mut s);
        size = 0;
        self.word2idx = words.into_iter().filter(|&(_, ref v)| v.1 >= 5).collect();
        for (_, v) in self.word2idx.iter_mut() {
            v.0 = size;
            size += 1;
        }
        println!("\r Read {} M words", self.ntokens / 1000000);
        let mut s = String::new();
        io::stdin().read_to_string(&mut s);
    }
}
