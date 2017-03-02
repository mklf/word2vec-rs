use std::ffi::{CStr,CString};
use std::os::raw::c_char;
use Argument;
use parse_arguments;
use train;
use std::error::Error;
// 0 : ok with ptr filled with argument 
// -1: utf8 unicode parse error
//-2 : 


#[no_mangle]
pub extern "C" fn ffi_train(
    args:*const *const c_char,n:i32,ptr: *mut *mut c_char
    )->i32{
    let mut v: Vec<String> = vec![];
    for i in 0..n as isize{
        unsafe{
            let s = CStr::from_ptr(*(args.offset(i))).to_str();
            match s {
                Ok(arg) => v.push(arg.to_string()),
                Err(e)=> {
                    let reason = CString::from_vec_unchecked(
                        e.description().as_bytes().to_vec()
                        );
                    *ptr = reason.into_raw();
                    return -1;
                },
            }
        }
    }
    let args = parse_arguments(&v).unwrap();
    let w2v = train(&args);
    match w2v {
        Ok(model) => {
            match model.save_vectors(&args.output){
                Ok(_) => {},
                // save error
                Err(e) => {
                    unsafe{
                        let reason = CString::from_vec_unchecked(
                            e.description().as_bytes().to_vec()
                            );
                        *ptr = reason.into_raw();
                    }
                   return -1
                }
            }
             
        },
        // train error
        Err(e) => {
            unsafe{
             let reason = CString::from_vec_unchecked(
                        format!("{}",e).as_bytes().to_vec()
                        );
                    *ptr = reason.into_raw();
            }
            return -1;

        },
    }  
    // no error
    0
}