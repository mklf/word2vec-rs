use std::io::{BufReader, stdout, stderr};
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;
use std::process;
use NEGATIVE_TABLE_SIZE;
use rand::{thread_rng, Rng};
pub struct Dict {
    word2ent: HashMap<String, Entry>,
    pub idx2word: Vec<String>,
    pub ntokens: usize,
    size: usize,
}
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
        }
    }
    pub fn init_negative_table(&self) -> Vec<usize> {
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
        negative_table

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

    pub fn counts(&self) -> Vec<u32> {

        let mut counts_ = vec![0;self.idx2word.len()];
        for (i, v) in self.idx2word.iter().enumerate() {
            counts_[i] = self.word2ent[v].count;
        }
        counts_
    }
    pub fn read_line(&self, line: &mut String, lines: &mut Vec<usize>) -> usize {
        let mut i = 0;
        for word in line.split_whitespace() {
            i += 1;
            match self.word2ent.get(word) {
                Some(e) => {
                    lines.push(e.index);
                }
                None => {}
            }
        }
        i
    }
    pub fn new_from_file(filename: &str, min_count: u32) -> Dict {
        let mut dict = Dict::new();
        let input_file = match File::open(filename) {
            Ok(fp) => fp,
            Err(e) => {
                stderr().write_fmt(format_args!("{}[文件名:{}]\n", e, filename)).unwrap();
                process::exit(1);
            }
        };
        let mut reader = BufReader::with_capacity(10000, input_file);
        let mut buf_str = String::with_capacity(5000);
        let mut words: HashMap<String, Entry> = HashMap::new();
        let (mut ntokens, mut size) = (0, 0);
        while reader.read_line(&mut buf_str).unwrap() > 0 {
            for word in buf_str.split_whitespace() {
                Dict::add_to_dict(&mut words, word, &mut size);
                ntokens += 1;
                if ntokens % 1000000 == 0 {
                    print!("\rRead {}M words", ntokens / 1000000);
                    stdout().flush().ok().expect("Could not flush stdout");
                }
            }
            buf_str.clear();
        }
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
        println!("\r Read {} M words", ntokens / 1000000);
        println!("\r {} unique words in total", size);
        dict
    }
}
