#[macro_use]
extern crate rustacuda;

use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::{error::Error};
use std::ffi::CString;
use rand::{RngCore, thread_rng};

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("./aes.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const size:usize = 16*1;
    // Allocate space on the device and copy numbers to it.
    let mut source: [u8;size] = [0;size];
    let mut rng = thread_rng();
    rng.fill_bytes(&mut source[..]);
    let mut input = DeviceBuffer::from_slice(&source)?;

    let mut key_slice: [u8;5] = [0;5];
    rng.fill_bytes(&mut key_slice);
    let mut key: DeviceBuffer<u8> = DeviceBuffer::from_slice(&key_slice)?;

    let mut keyLen = DeviceBox::new(&5i32)?;
    let mut blockNumber = DeviceBox::new(&1i32)?;


    let mut result_bytes: [u8;size] = [0;size];
    let mut result = DeviceBuffer::from_slice(&result_bytes)?;

    println!("key is {:?}",  hex::encode(&key_slice));
    println!("input is {:?}",  hex::encode(&source));
    //es_block aes_block_array[], BYTE key[], int keyLen, int block_number

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        /*
           grid_size: G,
        block_size: B,
        shared_mem_bytes: u32,
        args: &[*mut c_void],
         */

        //grid_size 和 block_size 分别代表了本次 kernel 启动对应的 block 数量和每个 block 中 thread 的数量，所以显然两者都要大于 0。
        // Launch the `sum` function with one block containing one thread on the given stream.
        launch!(module.AES_Encrypt<<<1, 1, 0, stream>>>(
           5,
           1,
            input.as_device_ptr(),
            key.as_device_ptr(),
            result.as_device_ptr(),
            size // Length
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;


    println!("input is {:?}",  hex::encode(&source));

    result.copy_to(&mut result_bytes)?;
    println!("output is {:?}",  hex::encode(&result_bytes));

    drop(result);
    Ok(())
}