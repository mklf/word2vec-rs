use dictionary::Dict;
use matrix::Matrix;
use std::thread;
use std;
use std::io::{BufReader, BufRead};
use std::fs::File;
extern crate rand;
use self::rand::distributions::{IndependentSample, Range};
use self::rand::Rng;
const SIGMOID_TABLE_SIZE: usize = 512;
const MAX_SIGMOID: f32 = 8f32;
const NEGATIVE_TABLE_SIZE: usize = 10000000;

pub struct Model {
    negative: usize,
    window: usize,
    embedding: Vec<f32>,
    vector_size: usize,
    dict: Dict,
    sigmoid_table: [f32; SIGMOID_TABLE_SIZE],
    negative_table: Vec<usize>,
}
pub struct ModelBuilder {
    negative: usize,
    window: usize,
    vector_size: usize,
}
impl ModelBuilder {
    pub fn new() -> ModelBuilder {
        ModelBuilder {
            negative: 5,
            window: 5,
            vector_size: 100,
        }
    }
    pub fn negative(&mut self, negative: usize) -> &mut ModelBuilder {
        self.negative = negative;
        self
    }
    pub fn window(&mut self, window: usize) -> &mut ModelBuilder {
        self.window = window;
        self
    }
    pub fn vector_size(&mut self, vector_size: usize) -> &mut ModelBuilder {
        self.vector_size = vector_size;
        self
    }
    pub fn finallize(&self) -> Model {
        Model {
            negative: self.negative,
            window: self.window,
            vector_size: self.vector_size,
            embedding: Vec::new(),
            dict: Dict::new(),
            sigmoid_table: [0f32; SIGMOID_TABLE_SIZE],
            negative_table: Vec::new(),
        }
    }
}
impl Model {
    fn init_embedding(&mut self, nrows: usize) {
        self.embedding = vec![0f32;self.vector_size * nrows];
        let between = Range::new(-1f32, 1.);
        let mut rng = rand::thread_rng();
        for v in &mut self.embedding {
            *v = between.ind_sample(&mut rng);
        }
    }
    fn skipgram(i: u32, matrix: &mut Vec<f32>) {
        matrix[i as usize] = i as f32;
    }
    fn init_sigmoid_table(&mut self) {
        for i in 0..SIGMOID_TABLE_SIZE {
            let x = (i as f64 * 2f64 * MAX_SIGMOID as f64) / SIGMOID_TABLE_SIZE as f64 -
                    MAX_SIGMOID as f64;
            self.sigmoid_table[i] = 1.0 / (1.0 + (-x).exp()) as f32;
        }
    }
    fn init_negative_table(&mut self) {
        let counts = self.dict.counts();
        let mut z = 0f64;
        for c in &counts {
            z += (*c as f64).powf(0.5);
        }
        for i in counts {
            let c = (i as f64).powf(0.5);
            for j in 0..(c * NEGATIVE_TABLE_SIZE as f64 / z) as usize {
                self.negative_table.push(i);
            }
        }
        let length = self.negative_table.len();
        for i in 0..self.negative_table.len() {
            let idx: usize = (rand::thread_rng().next_u32() % length as u32) as usize;
            let tmp = self.negative_table[i];
            self.negative_table[i] = self.negative_table[idx];
            self.negative_table[idx as usize] = tmp;
        }

    }


    pub fn train(&mut self, filename: &str, workers: u32) {
        let mut dict = Dict::new();
        dict.read_from_file(filename);
        self.init_embedding(dict.nsize());
        self.dict = dict;
        self.init_sigmoid_table();
        self.init_negative_table();
        let mut handles: Vec<_> = Vec::new();
        let mut mo = &mut *self as *mut Model;

        unsafe {
            for i in 0..workers {
                let model = std::mem::transmute::<*mut Model, u64>(mo);
                let filename = filename.to_string();
                handles.push(thread::spawn(move || {
                    let ref mut model = *std::mem::transmute::<u64, *mut Model>(model);
                    let ref dict = model.dict;
                    let input_file = File::open(filename).unwrap();
                    let mut reader = BufReader::with_capacity(10000, input_file);
                    let mut lines = Vec::with_capacity(1000);
                    let mut mat =
                        Matrix::new(&mut model.embedding, dict.nsize(), model.vector_size);
                    for line in reader.lines() {
                        let mut line = line.unwrap();
                        dict.read_line(&mut line, &mut lines);
                        // Model::skipgram(i, &mut model.embedding);
                        lines.clear();
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        }
    }
}