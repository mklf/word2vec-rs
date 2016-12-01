use std::io::{BufReader, stdout};
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;
use NEGATIVE_TABLE_SIZE;
use rand::{thread_rng, Rng};
use rand::distributions::{IndependentSample, Range};
use std::sync::Arc;
use super::W2vError;
#[derive(RustcEncodable, RustcDecodable, PartialEq,Debug)]
pub struct Dict {
    word2ent: HashMap<String, Entry>,
    pub idx2word: Vec<String>,
    pub ntokens: usize,
    size: usize,
    discard_table: Vec<f32>,
}

#[derive(RustcEncodable, RustcDecodable, PartialEq,Debug)]
struct Entry {
    index: usize,
    count: u32,
}


impl Dict {
    fn new() -> Dict {
        Dict {
            word2ent: HashMap::new(),
            idx2word: Vec::new(),
            ntokens: 0,
            size: 0,
            discard_table: Vec::new(),
        }
    }
    pub fn init_negative_table(&self) -> Arc<Vec<usize>> {
        let mut negative_table = Vec::new();
        let counts = self.counts();
        let mut z = 0f64;
        for c in &counts {
            z += (*c as f64).powf(0.5);
        }
        for (idx, i) in counts.into_iter().enumerate() {
            let c = (i as f64).powf(0.5);
            for _ in 0..(c * NEGATIVE_TABLE_SIZE as f64 / z) as usize {
                negative_table.push(idx as usize);
            }
        }
        let mut rng = thread_rng();
        rng.shuffle(&mut negative_table);
        Arc::new(negative_table)

    }

    fn add_to_dict(words: &mut HashMap<String, Entry>, word: &str, size: &mut usize) {
        words.entry(word.to_owned())
            .or_insert_with(|| {
                let ent = Entry {
                    index: *size,
                    count: 0,
                };
                *size += 1;
                ent
            })
            .count += 1;
    }
    #[inline(always)]
    pub fn nsize(&self) -> usize {
        self.size
    }
    #[inline(always)]
    pub fn get_idx(&self, word: &str) -> usize {
        self.word2ent[word].index
    }
    #[inline(always)]
    pub fn get_word(&self, idx: usize) -> String {
        self.idx2word[idx].clone()
    }

    pub fn counts(&self) -> Vec<u32> {
        let mut counts_ = vec![0;self.idx2word.len()];
        for (i, v) in self.idx2word.iter().enumerate() {
            counts_[i] = self.word2ent[v].count;
        }
        counts_
    }

    pub fn convert_line(&self, line: &String, lines: &mut Vec<usize>) -> usize {
        let mut i = 0;
        let mut rng = thread_rng();
        let between = Range::new(0., 1.);
        for word in line.split_whitespace() {
            i += 1;
            match self.word2ent.get(word) {
                Some(e) => {
                    if self.discard_table[e.index] > between.ind_sample(&mut rng) {
                        lines.push(e.index);
                    }
                }
                None => {}
            }
        }
        i
    }
    #[inline]
    fn add_line(mut words: &mut HashMap<String, Entry>,
                buffer: Vec<u8>,
                ntokens: &mut usize,
                size: &mut usize,
                verbose: bool) {

        let line = String::from_utf8_lossy(&buffer);
        for w in line.split_whitespace() {
            Dict::add_to_dict(words, w, size);
            *ntokens += 1;
            if verbose && *ntokens % 1000000 == 0 {
                print!("\rRead {}M words", *ntokens / 1000000);
                stdout().flush().ok().expect("Could not flush stdout");
            }
        }
    }
    pub fn new_from_file(filename: &str,
                         min_count: u32,
                         threshold: f32,
                         verbose: bool)
                         -> Result<Dict, W2vError> {
        let mut dict = Dict::new();
        let input_file = try!(File::open(filename));
        let reader = BufReader::with_capacity(10000, input_file);
        let mut words: HashMap<String, Entry> = HashMap::new();
        let (mut ntokens, mut size) = (0, 0);
        let mut buf_vec = vec![0; 10000];
        for ch in reader.bytes() {
            let c = ch.unwrap();
            if buf_vec.len() > 10000 && c == b' ' {
                Dict::add_line(&mut words,
                               buf_vec.clone(),
                               &mut ntokens,
                               &mut size,
                               verbose);
                buf_vec.clear();
            }
            buf_vec.push(c);
        }

        Dict::add_line(&mut words, buf_vec, &mut ntokens, &mut size, verbose);

        size = 0;
        let word2ent: HashMap<String, Entry> = words.into_iter()
            .filter(|&(_, ref v)| v.count >= min_count)
            .map(|(k, mut v)| {
                v.index = size;
                size += 1;
                (k, v)
            })
            .collect();
        dict.word2ent = word2ent;
        dict.word2ent.shrink_to_fit();
        dict.idx2word = vec!["".to_string();dict.word2ent.len()];
        for (k, v) in &dict.word2ent {
            dict.idx2word[v.index] = k.to_string();
        }
        dict.idx2word.shrink_to_fit();
        dict.size = size;
        dict.ntokens = ntokens;
        if verbose {
            println!("\rRead {} M words", (ntokens / 1000000));
            println!("\r{} unique words in total", size);

        }
        dict.init_discard(threshold);
        Ok(dict)
    }
    fn init_discard(&mut self, threshold: f32) {
        let size = self.nsize();
        self.discard_table.reserve_exact(size);
        for i in 0..self.nsize() {
            let f = self.word2ent[&self.idx2word[i]].count as f32 / self.ntokens as f32;
            self.discard_table.push((threshold / f).sqrt() + threshold / f);
        }
    }
}