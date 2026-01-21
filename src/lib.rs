use std::fs::File;
use std::path::Path;
use std::borrow::Cow;

use pyo3::prelude::*;
use memmap2::MmapOptions;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// -----------------------------------------------------------------------------
// 1. Data Structures
// -----------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GaussianSplat {
    pub pos: [f32; 3],      // 12
    pub opacity: f32,       // 4
    pub scale: [f32; 3],    // 12
    pub _pad1: f32,         // 4
    pub rot: [f32; 4],      // 16
    pub sh_dc: [f32; 3],    // 12
    pub _pad2: f32,         // 4
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Surfel {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub cov_row0: [f32; 3],
    pub _pad2: f32,
    pub cov_row1: [f32; 3],
    pub _pad3: f32,
    pub cov_row2: [f32; 3],
    pub _pad4: f32,
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct RawSplat {
    x: f32, y: f32, z: f32,
    nx: f32, ny: f32, nz: f32,
    f_dc_0: f32, f_dc_1: f32, f_dc_2: f32,
    opacity: f32,
    scale_0: f32, scale_1: f32, scale_2: f32,
    rot_0: f32, rot_1: f32, rot_2: f32, rot_3: f32,
}

// -----------------------------------------------------------------------------
// 2. GPU Logic
// -----------------------------------------------------------------------------

async fn run_compute_shader(splats: &[GaussianSplat]) -> Vec<Surfel> {
    // 1. インスタンス作成: 全てのバックエンドを許可 (Vulkan, GL, DX12, Metal)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        // backends: wgpu::Backends::all(), 
        backends: wgpu::Backends::VULKAN,
        // backends: wgpu::Backends::PRIMARY, 
        ..Default::default()
    });
    
    // -------------------------------------------------------------------------
    // 2. アダプターの列挙と選択 (デバッグ表示付き)
    // -------------------------------------------------------------------------
    println!("[GPU Debug] Enumerating Vulkan adapters...");
    let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    //let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

    if adapters.is_empty() {
        println!("[GPU Error] No Vulkan adapters found!");
        println!("[GPU Hint] WSL2? Try: sudo apt install mesa-vulkan-drivers vulkan-tools");
        panic!("No Vulkan-compatible GPU found.");
    }

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("  [{}] {:?} ({:?})", i, info.name, info.backend);
    }

    // 最も高性能なアダプタを要求
    // let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
    //     power_preference: wgpu::PowerPreference::HighPerformance,
    //     compatible_surface: None,
    //     force_fallback_adapter: false,
    // })
    // .await
    // .expect("Failed to select a Vulkan adapter.");
    let adapter = adapters.into_iter()
        .filter(|a| {
            let info = a.get_info();
            // OpenGL系は除外
            info.backend != wgpu::Backend::Gl && info.backend != wgpu::Backend::BrowserWebGpu
        })
        .max_by_key(|a| {
            // 優先順位: Vulkan > DX12 > Metal > その他
            let info = a.get_info();
            match info.backend {
                wgpu::Backend::Vulkan => 3,
                wgpu::Backend::Dx12 => 2,
                wgpu::Backend::Metal => 2,
                _ => 1,
            }
        });

    let adapter = match adapter {
        Some(a) => a,
        None => panic!("[GPU Error] Adapters found, but all were OpenGL (which crashes WSL2). Enable Vulkan or DX12!"),
    };

    let info = adapter.get_info();
    println!("[GPU Debug] Selected: {:?} ({:?})", info.name, info.backend);

    // -------------------------------------------------------------------------
    // 3. デバイスとキューの取得
    // -------------------------------------------------------------------------
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ).await.expect("Failed to create device");

    // -------------------------------------------------------------------------
    // 4. バッファ・シェーダー・パイプライン
    // -------------------------------------------------------------------------
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(splats),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_size = (splats.len() * std::mem::size_of::<Surfel>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. パイプラインの構築
    let shader_source = include_str!("shader.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
    });

    // 4. コマンドの発行
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        let workgroups = (splats.len() as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // 5. 結果の取得
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);
    
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<Surfel> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    } else {
        panic!("Failed to receive buffer map result");
    }
}

// -----------------------------------------------------------------------------
// 3. Python Interface
// -----------------------------------------------------------------------------

#[pyclass]
struct SplatManager {
    splats: Vec<GaussianSplat>,
    surfels: Vec<Surfel>, // 計算結果キャッシュ
}

#[pymethods]
impl SplatManager {
    #[new]
    fn new(ply_path: String) -> PyResult<Self> {
        // ... (Phase 1 と同じ読み込みコード) ...
        let path = Path::new(&ply_path);
        let file = File::open(path).map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(e.to_string()))?;
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? };

        let header_end_pattern = b"end_header";
        let search_limit = std::cmp::min(mmap.len(), 4096);
        let data_start_offset = if let Some(pos) = mmap[..search_limit].windows(header_end_pattern.len()).position(|w| w == header_end_pattern) {
            let mut cursor = pos + header_end_pattern.len();
            while cursor < mmap.len() && (mmap[cursor] == b'\r' || mmap[cursor] == b'\n') { cursor += 1; }
            cursor
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid PLY"));
        };

        let raw_data = &mmap[data_start_offset..];
        let struct_size = std::mem::size_of::<RawSplat>();
        let count = raw_data.len() / struct_size;
        let mut splats = Vec::with_capacity(count);
        let raw_splats: &[RawSplat] = bytemuck::cast_slice(&raw_data[..count * struct_size]);

        for raw in raw_splats {
            splats.push(GaussianSplat {
                pos: [raw.x, raw.y, raw.z],
                opacity: raw.opacity,
                scale: [raw.scale_0, raw.scale_1, raw.scale_2],
                _pad1: 0.0,
                rot: [raw.rot_0, raw.rot_1, raw.rot_2, raw.rot_3],
                sh_dc: [raw.f_dc_0, raw.f_dc_1, raw.f_dc_2],
                _pad2: 0.0,
            });
        }

        Ok(SplatManager { splats, surfels: Vec::new() })
    }

    fn count(&self) -> usize { self.splats.len() }

    fn debug_first_splat(&self) -> PyResult<String> {
        if let Some(s) = self.splats.first() {
            Ok(format!(
                "Pos: [{:.4}, {:.4}, {:.4}], Opacity: {:.4}, Scale: [{:.4}, ...]",
                s.pos[0], s.pos[1], s.pos[2],
                s.opacity,
                s.scale[0]
            ))
        } else {
            Ok("No splats loaded".to_string())
        }
    }

    /// GPUコンピュートを実行し、法線と共分散行列を計算する
    fn compute_geometry(&mut self) -> PyResult<usize> {
        if self.splats.is_empty() { return Ok(0); }
        
        // pollster::block_on で非同期関数を同期的に実行
        self.surfels = pollster::block_on(run_compute_shader(&self.splats));
        
        Ok(self.surfels.len())
    }

    /// デバッグ用：最初のサーフェル情報を取得
    fn debug_first_surfel(&self) -> PyResult<String> {
        if let Some(s) = self.surfels.first() {
            Ok(format!(
                "Normal: [{:.4}, {:.4}, {:.4}], Cov[0]: [{:.4}, ...]",
                s.normal[0], s.normal[1], s.normal[2],
                s.cov_row0[0]
            ))
        } else {
            Ok("No surfels computed. Run compute_geometry() first.".to_string())
        }
    }
}

#[pymodule]
fn gs_slam_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplatManager>()?;
    Ok(())
}
