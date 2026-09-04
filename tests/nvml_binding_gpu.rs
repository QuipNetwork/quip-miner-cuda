//! GPU-gated checks that the utilization governor binds the same physical GPU
//! the CUDA context opened.
//!
//! Ignored by default: these need a real CUDA device. Run with
//! `cargo test --test nvml_binding_gpu -- --ignored`.
//!
//! The bug these guard (quip-miner-cuda-dot) is silent by construction: NVML
//! enumerates in PCI bus order while a CUDA ordinal follows
//! `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES`, so an ordinal handed to
//! `device_by_index` yields a working governor that measures the wrong card.
//! Only a per-device identity comparison catches it.

use quip_miner_cuda::cuda_device::CudaDevice;

/// The address CUDA reports must be the address NVML knows the same card by.
/// Without this, `Nvml::device_by_pci_bus_id` silently finds nothing and the
/// governor never throttles.
#[test]
#[ignore = "requires a CUDA GPU"]
fn every_cuda_device_resolves_in_nvml_by_its_bus_id() {
    let count = CudaDevice::device_count().unwrap_or(0);
    if count == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }
    let nvml = nvml_wrapper::Nvml::init().expect("NVML init");

    for index in 0..count {
        let device = CudaDevice::open(index).expect("open CUDA device");
        let handle = nvml
            .device_by_pci_bus_id(device.pci_bus_id.clone())
            .unwrap_or_else(|e| panic!("NVML has no device at {}: {e}", device.pci_bus_id));
        // NVML echoes the address back in its own canonical form; a round trip
        // that changes the string would mean the format is only accidentally
        // accepted.
        let echoed = handle.pci_info().expect("NVML pci info").bus_id;
        assert_eq!(
            echoed, device.pci_bus_id,
            "CUDA ordinal {index} and NVML disagree on the bus id"
        );
    }
}

/// Distinct CUDA ordinals must be distinct physical cards. On a single-GPU
/// host this asserts nothing; on a multi-GPU host it is what proves
/// `--device 1` reaches a second GPU rather than aliasing the first.
#[test]
#[ignore = "requires a CUDA GPU"]
fn distinct_ordinals_name_distinct_cards() {
    let count = CudaDevice::device_count().unwrap_or(0);
    if count < 2 {
        eprintln!("fewer than two CUDA devices visible; skipping");
        return;
    }
    let first = CudaDevice::open(0).expect("open device 0").pci_bus_id;
    let second = CudaDevice::open(1).expect("open device 1").pci_bus_id;
    assert_ne!(first, second, "device 0 and device 1 share a bus id");
}
