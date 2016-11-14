pub struct Matrix<'a> {
    nrows: usize,
    ncols: usize,
    mat: &'a mut Vec<f32>,
}
impl<'a> Matrix<'a> {
    pub fn new(emb: &'a mut Vec<f32>, ncols: usize, nrows: usize) -> Matrix<'a> {
        Matrix {
            mat: emb,
            ncols: ncols,
            nrows: nrows,
        }
    }
    #[inline(always)]
    pub fn zero(&mut self) {
        for v in &mut self.mat.iter_mut() {
            *v = 0f32;
        }
    }
    #[inline(always)]
    pub fn add_row(&mut self, vec: &Vec<f32>, i: usize, mul: f32) {

        for t in 0..self.nrows {
            unsafe {
                *self.mat.get_unchecked_mut(i * self.nrows + t) += mul * (*vec.get_unchecked(t));
            }
        }
    }
    #[inline(always)]
    pub fn dot_row(&mut self, vec: &Vec<f32>, i: usize) -> f32 {
        let mut sum = 0f32;
        for t in 0..self.nrows {
            unsafe {
                sum += *self.mat.get_unchecked(i * self.nrows + t) * (*vec.get_unchecked(t));
            };
        }
        sum
    }
}