import ctypes
try:
    lib = ctypes.CDLL("../target/release/libswsg.dylib")
except OSError:
    lib = ctypes.CDLL("../target/release/swsg.so")
func = lib.ffi_train

args = "dummy train CORPUS  SAVE_DIR --thread 12 --epoch 5 --verbose --neg 5 --min_count 5  --lr 0.05".split()


arguments_type= (ctypes.c_char_p * len(args))
func.argtypes=[arguments_type,ctypes.c_int32,ctypes.POINTER(ctypes.c_char_p)]
arr = arguments_type()
arr[:] = [_.encode() for _ in args]
reason = ctypes.c_char_p()
print(func(arr,len(args),ctypes.byref(reason)))
print (reason.value)