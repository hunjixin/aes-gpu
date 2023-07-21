#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate space on the device and copy numbers to it.
    let mut x = DeviceBox::new(&10.0f32)?;
    let mut y = DeviceBox::new(&20.0f32)?;
    let mut result = DeviceBox::new(&0.0f32)?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        // Launch the `sum` function with one block containing one thread on the given stream.
        launch!(module.sum<<<1, 1, 0, stream>>>(
            x.as_device_ptr(),
            y.as_device_ptr(),
            result.as_device_ptr(),
            1 // Length
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    println!("Sum is {}", result_host);

    Ok(())
}