use dictionary::Dict;
use std::thread;
use std;
extern crate rand;
use self::rand::distributions::{IndependentSample, Range};
pub struct Model {
    negative: usize,
    window: usize,
    embedding: Vec<f32>,
    vector_size: usize,
    dict: Dict,
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
    pub fn train(mut self, filename: &str, workers: u32) {
        let mut dict = Dict::new();
        dict.read_from_file(filename);
        self.init_embedding(dict.nsize());

        let mut s = String::new();
        println!("type to exit");
        std::io::stdin().read_line(&mut s).unwrap();
        let mut handles: Vec<_> = Vec::new();
        let emb_data = &mut self.embedding as *mut Vec<_>;

        unsafe {
            for i in 0..workers {
                let x = std::mem::transmute::<*mut Vec<f32>, u64>(emb_data);
                handles.push(thread::spawn(move || {
                    let ref k = *std::mem::transmute::<u64, *mut Vec<f32>>(x);
                    print!("{} ", k[1]);
                }))
            }
        }
        for h in handles {
            h.join();
        }
        self.dict = dict;
    }
}