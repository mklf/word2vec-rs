use matrix::Matrix;
extern crate rand;
use rand::{ThreadRng, thread_rng};
use {MAX_SIGMOID, SIGMOID_TABLE_SIZE, LOG_TABLE_SIZE};



fn init_sigmoid_table() -> [f32; SIGMOID_TABLE_SIZE + 1] {
    let mut sigmoid_table = [0f32; SIGMOID_TABLE_SIZE + 1];
    for i in 0..SIGMOID_TABLE_SIZE + 1 {
        let x = (i as f64 * 2f64 * MAX_SIGMOID as f64) / SIGMOID_TABLE_SIZE as f64 -
                MAX_SIGMOID as f64;
        sigmoid_table[i] = 1.0 / (1.0 + (-x).exp()) as f32;
    }
    sigmoid_table
}
fn init_log_table() -> [f32; LOG_TABLE_SIZE + 1] {
    let mut log_table = [0f32; LOG_TABLE_SIZE + 1];
    for i in 0..LOG_TABLE_SIZE + 1 {
        let x = (i as f32 + 1e-5) / LOG_TABLE_SIZE as f32;
        log_table[i] = x.ln();
    }
    log_table
}


pub struct Model<'a> {
    input: &'a mut Matrix,
    output: &'a mut Matrix,
    dim: usize,
    lr: f32,
    neg: usize,
    grad_: Vec<f32>,
    neg_pos: usize,
    sigmoid_table: [f32; SIGMOID_TABLE_SIZE + 1],
    log_table: [f32; LOG_TABLE_SIZE + 1],
    negative_table: Vec<usize>,
    loss: f32,
    nsamples: usize,
}
impl<'a> Model<'a> {
    pub fn new(input: &'a mut Matrix,
               output: &'a mut Matrix,
               dim: usize,
               lr: f32,
               tid: u32,
               neg: usize,
               neg_table: Vec<usize>)
               -> Model<'a> {
        Model {
            input: input,
            output: output,
            dim: dim,
            lr: lr,
            neg: neg,
            grad_: vec![0f32;dim],
            neg_pos: 0,
            sigmoid_table: init_sigmoid_table(),
            log_table: init_log_table(),
            negative_table: neg_table,
            loss: 0.,
            nsamples: 0,
        }

    }
    #[inline]
    fn log(&self, x: f32) -> f32 {
        if x > 1.0 {
            return x;
        }
        let i = (x * (LOG_TABLE_SIZE as f32)) as usize;
        self.log_table[i]
    }
    #[inline]
    fn sigmoid(&self, x: f32) -> f32 {
        if x < -MAX_SIGMOID {
            0f32
        } else if x > MAX_SIGMOID {
            1f32
        } else {
            let i = (x + MAX_SIGMOID as f32) * SIGMOID_TABLE_SIZE as f32 / MAX_SIGMOID as f32 / 2.;
            self.sigmoid_table[i as usize]
        }
    }
    #[inline]
    pub fn get_loss(&self) -> f32 {
        self.loss / self.nsamples as f32
    }

    #[inline(always)]
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    #[inline(always)]
    pub fn get_lr(&self) -> f32 {
        self.lr
    }

    fn binary_losgistic(&mut self, input_emb: *mut f32, target: usize, label: bool) -> f32 {
        let sum = self.output.dot_row(input_emb, target);
        let score = self.sigmoid(sum);
        let alpha = self.lr * (label as i32 as f32 - score);
        let tar_emb = self.output.get_row(target);
        self.add_mul_row(tar_emb, alpha);
        self.output.add_row(input_emb, target, alpha);
        if label {
            -self.log(score)
        } else {
            -self.log(1.0 - score)
        }
    }
    #[inline(always)]
    pub fn update(&mut self, input: usize, target: usize) {
        self.loss += self.negative_sampling(input, target);
        self.nsamples += 1;
    }

    fn negative_sampling(&mut self, input: usize, target: usize) -> f32 {
        let input_emb = self.input.get_row(input);
        let mut loss = 0f32;
        self.grad_zero();
        for i in 0..self.neg {
            if i == 0 {
                loss += self.binary_losgistic(input_emb, target, true);
            } else {
                let neg_sample = self.get_negative(target);
                loss += self.binary_losgistic(input_emb, neg_sample, false);
            }
        }
        self.input.add_row(self.grad_.as_mut_ptr(), input, 1.0);
        loss
    }
    fn get_negative(&mut self, target: usize) -> usize {
        loop {
            let negative = self.negative_table[self.neg_pos];
            self.neg_pos = (self.neg_pos + 1) % self.negative_table.len();
            if target != negative {
                return negative;
            }
        }
    }

    #[inline(always)]
    fn grad_zero(&mut self) {
        for a in self.grad_.as_mut_slice() {
            *a = 0f32;
        }
    }
    #[inline(always)]
    fn add_mul_row(&mut self, other: *mut f32, a: f32) {
        for i in 0..self.grad_.len() {
            unsafe {
                *self.grad_.get_unchecked_mut(i) += a * (*other.offset(i as isize));
            }
        }
    }
}