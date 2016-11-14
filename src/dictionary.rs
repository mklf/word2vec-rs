use std::io::{BufReader, stdout};
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;

const MAX_VOCAB_SIZE: usize = 30000000;
const MAX_LINE_LENGHT: usize = 1000;


pub struct Dict {
    word2ent: HashMap<String, Entry>,
    idx2word: Vec<String>,
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

impl Dict {
    pub fn new() -> Dict {
        Dict {
            word2ent: HashMap::new(),
            idx2word: Vec::new(),
            ntokens: 0,
            size: 0,
        }
    }
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
    pub fn nsize(&self) -> usize {
        self.size
    }
    pub fn counts(&self) -> Vec<usize> {

        let mut counts_ = vec![0;self.idx2word.len()];
        for (i, v) in self.idx2word.iter().enumerate() {
            counts_[i] = self.word2ent[v].index;
        }
        counts_
    }
    pub fn read_line(&self, line: &mut String, lines: &mut Vec<usize>) {
        for word in line.split_whitespace() {
            match self.word2ent.get(word) {
                Some(e) => {
                    lines.push(e.index);
                }
                None => {}
            }
        }
    }
    pub fn read_from_file(&mut self, filename: &str) {
        let input_file = File::open(filename).unwrap();
        let mut reader = BufReader::with_capacity(10000, input_file);
        let mut buf_str = String::with_capacity(5000);
        let mut words: HashMap<String, Entry> = HashMap::new();
        let (mut ntokens, mut size) = (0, 0);
        while reader.read_line(&mut buf_str).unwrap() > 0 {
            for word in buf_str.split_whitespace() {
                Dict::add_to_dict(&mut words, word, &mut size);
                ntokens += 1;
                if ntokens % 1000000 == 0 {
                    print!("\r read {}M words", ntokens / 1000000);
                    stdout().flush().ok().expect("Could not flush stdout");
                }
            }
            buf_str.clear();
        }
        size = 0;
        let word2ent: HashMap<String, Entry> = words.into_iter()
            .filter(|&(_, ref v)| v.count >= 5)
            .map(|(k, mut v)| {
                v.index = size;
                size += 1;
                (k, v)
            })
            .collect();
        self.word2ent = word2ent;
        self.word2ent.shrink_to_fit();
        self.idx2word = vec!["".to_string();self.word2ent.len()];
        for (k, v) in &self.word2ent {
            self.idx2word[v.index] = k.to_string();
        }
        self.idx2word.shrink_to_fit();
        self.size = size;
        self.ntokens = ntokens;
        println!("\r Read {} M words", ntokens / 1000000);
        println!("\r {} unique words in total", size);
    }
}
