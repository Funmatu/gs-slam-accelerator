use std::borrow::Cow;
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "wasm")]
use wgpu::util::DeviceExt;

#[cfg(any(feature = "python", feature = "wasm"))]
use nalgebra as na;

// ============================================================================
//  1. Data Structures (16-byte Aligned for WebGPU Compatibility)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GaussianSplat {
    // pos(xyz), opacity(w) -> 16 bytes
    pub pos: [f32; 3],
    pub opacity: f32,
    
    // scale(xyz), pad(w) -> 16 bytes
    pub scale: [f32; 3],
    pub _pad1: f32,
    
    // rot(xyzw) -> 16 bytes
    pub rot: [f32; 4],
    
    // sh(xyz), pad(w) -> 16 bytes
    pub sh_dc: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Surfel {
    // pos(xyz), pad(w) -> 16 bytes
    pub pos: [f32; 3],
    pub _pad0: f32,
    
    // color(xyz), pad(w) -> 16 bytes
    pub color: [f32; 3],
    pub _pad1: f32,
    
    // normal(xyz), pad(w) -> 16 bytes
    pub normal: [f32; 3],
    pub _pad2: f32,
}

// Input PLY format (Packed)
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

// ============================================================================
//  2. Core Logic (CPU Math Helpers)
// ============================================================================

// SH to RGB (0th order)
fn sh_to_rgb_cpu(sh: [f32; 3]) -> [f32; 3] {
    let c0 = 0.2820947917;
    [
        (0.5 + c0 * sh[0]).clamp(0.0, 1.0),
        (0.5 + c0 * sh[1]).clamp(0.0, 1.0),
        (0.5 + c0 * sh[2]).clamp(0.0, 1.0),
    ]
}

// Compute Normal from Rotation quaternion (x,y,z,w) and Scale
#[cfg(any(feature = "python", feature = "wasm"))]
fn compute_normal_cpu(rot: [f32; 4], scale: [f32; 3]) -> [f32; 3] {
    // rot is [x, y, z, w] stored in PLY/Struct
    // nalgebra expects w last for Quaternion constructor: (w, i, j, k)
    // Wait, typical PLY is w,x,y,z.
    // Our loader stores it as rot[0]..rot[3].
    // If input is (x,y,z,w), then q = new(w, x, y, z)
    let q = na::UnitQuaternion::new_normalize(na::Quaternion::new(rot[3], rot[0], rot[1], rot[2]));
    let r = q.to_rotation_matrix();
    
    // Axis with minimum scale
    let s_abs = [scale[0].abs(), scale[1].abs(), scale[2].abs()];
    let local_n = if s_abs[0] < s_abs[1] && s_abs[0] < s_abs[2] {
        na::Vector3::new(1.0, 0.0, 0.0)
    } else if s_abs[1] < s_abs[2] {
        na::Vector3::new(0.0, 1.0, 0.0)
    } else {
        na::Vector3::new(0.0, 0.0, 1.0)
    };

    let n = r * local_n;
    [n.x, n.y, n.z]
}

// ============================================================================
//  3. GPU Logic (Headless for Python)
// ============================================================================

#[cfg(feature = "python")]
async fn run_compute_shader_headless(splats: &[GaussianSplat]) -> Result<Vec<Surfel>, String> {
    use wgpu::util::DeviceExt;

    // 1. Instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    // 2. Adapter
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await.ok_or("Failed to find GPU adapter")?;

    // 3. Device
    // WSL2対策: required_limitsを下げておく
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), 
        memory_hints: wgpu::MemoryHints::default(),
    }, None).await.map_err(|e| format!("Failed to create device: {:?}", e))?;

    // 4. Buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(splats),
        usage: wgpu::BufferUsages::STORAGE,
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

    // 5. Pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: Some("compute_main"),
        compilation_options: Default::default(), cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
    });

    // 6. Execute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (splats.len() as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // 7. Readback
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<Surfel> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    } else {
        Err("Failed to map buffer".to_string())
    }
}

// ============================================================================
//  4. Python Module (PyO3)
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
struct SplatManager {
    splats: Vec<GaussianSplat>,
    surfels: Vec<Surfel>,
}

#[cfg(feature = "python")]
#[pymethods]
impl SplatManager {
    #[new]
    fn new(ply_path: String) -> PyResult<Self> {
        let path = std::path::Path::new(&ply_path);
        let file = std::fs::File::open(path).map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(e.to_string()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? };

        let header_end = b"end_header";
        let offset = mmap.windows(header_end.len())
            .position(|w| w == header_end)
            .map(|i| i + header_end.len())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid PLY header"))?;
        
        let mut cursor = offset;
        while cursor < mmap.len() && (mmap[cursor] == b'\r' || mmap[cursor] == b'\n') { cursor += 1; }
        
        let raw_data = &mmap[cursor..];
        let struct_size = std::mem::size_of::<RawSplat>();
        let count = raw_data.len() / struct_size;
        
        let raw_splats: &[RawSplat] = bytemuck::cast_slice(&raw_data[..count * struct_size]);
        let mut splats = Vec::with_capacity(count);

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

    // 基本情報
    fn count(&self) -> usize { self.splats.len() }

    // データアクセサ (テスト用)
    fn get_splat_pos(&self, idx: usize) -> PyResult<[f32; 3]> {
        self.splats.get(idx).map(|s| s.pos).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
    fn get_splat_rot(&self, idx: usize) -> PyResult<[f32; 4]> {
        self.splats.get(idx).map(|s| s.rot).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
    fn get_splat_sh(&self, idx: usize) -> PyResult<[f32; 3]> {
        self.splats.get(idx).map(|s| s.sh_dc).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }

    // GPU計算
    fn compute_geometry(&mut self) -> PyResult<usize> {
        if self.splats.is_empty() { return Ok(0); }
        match pollster::block_on(run_compute_shader_headless(&self.splats)) {
            Ok(res) => {
                self.surfels = res;
                Ok(self.surfels.len())
            },
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    // CPU計算 (Fallback)
    fn compute_geometry_cpu(&mut self) -> PyResult<usize> {
        self.surfels.clear();
        for s in &self.splats {
            let rgb = sh_to_rgb_cpu(s.sh_dc);
            let normal = compute_normal_cpu(s.rot, s.scale);
            
            self.surfels.push(Surfel {
                pos: s.pos, _pad0: 0.0,
                color: rgb, _pad1: 0.0,
                normal, _pad2: 0.0,
            });
        }
        Ok(self.surfels.len())
    }

    // 結果アクセサ
    fn get_surfel_color(&self, idx: usize) -> PyResult<[f32; 3]> {
        self.surfels.get(idx).map(|s| s.color).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
    fn get_surfel_normal(&self, idx: usize) -> PyResult<[f32; 3]> {
        self.surfels.get(idx).map(|s| s.normal).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn gs_slam_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplatManager>()?;
    Ok(())
}

// ============================================================================
//  5. WASM Viewer
// ============================================================================

#[cfg(feature = "wasm")]
struct CameraController {
    target: na::Point3<f32>,
    distance: f32,
    yaw: f32,
    pitch: f32,
    width: f32,
    height: f32,
}

#[cfg(feature = "wasm")]
impl CameraController {
    fn new(width: f32, height: f32) -> Self {
        Self {
            target: na::Point3::new(0.0, 0.0, 0.0),
            distance: 5.0,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            width,
            height,
        }
    }

    fn update_resolution(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch -= dy * 0.005;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let (view, _) = self.build_matrices();
        let right = na::Vector3::new(view[(0, 0)], view[(1, 0)], view[(2, 0)]);
        let up = na::Vector3::new(view[(0, 1)], view[(1, 1)], view[(2, 1)]);
        let speed = self.distance * 0.001;
        self.target -= right * dx * speed;
        self.target += up * dy * speed;
    }

    fn zoom(&mut self, delta: f32) {
        self.distance -= delta * self.distance * 0.001;
        self.distance = self.distance.clamp(0.1, 100.0);
    }

    fn build_matrices(&self) -> (na::Matrix4<f32>, na::Matrix4<f32>) {
        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();
        let eye = self.target + na::Vector3::new(x, y, z);
        let up = na::Vector3::y();
        let view = na::Matrix4::look_at_rh(&eye, &self.target, &up);
        let aspect = self.width / self.height;
        let proj = na::Matrix4::new_perspective(aspect, 45.0_f32.to_radians(), 0.1, 1000.0);
        (view, proj)
    }

    fn get_uniform(&self, mode: u32) -> [u8; 128] {
        let (view, proj) = self.build_matrices();
        #[rustfmt::skip]
        let correction = na::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        let view_proj = correction * proj * view;
        let vp_array: [[f32; 4]; 4] = view_proj.into();

        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();
        
        let mut raw = [0u8; 128];
        raw[0..64].copy_from_slice(bytemuck::cast_slice(&vp_array));
        raw[64..76].copy_from_slice(bytemuck::cast_slice(&[x, y, z]));
        // mode at offset 80
        raw[80..84].copy_from_slice(bytemuck::cast_slice(&[mode]));
        raw
    }
}

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys::{HtmlCanvasElement, MouseEvent, WheelEvent};
#[cfg(feature = "wasm")]
use std::rc::Rc;
#[cfg(feature = "wasm")]
use std::cell::RefCell;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmViewer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    bg_compute: Option<wgpu::BindGroup>,
    bg_render: Option<wgpu::BindGroup>,
    bgl_compute: wgpu::BindGroupLayout,
    bgl_render: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: Option<wgpu::Buffer>,
    num_vertices: u32,
    camera: Rc<RefCell<CameraController>>,
    display_mode: u32,
    _closures: Vec<wasm_bindgen::JsValue>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmViewer {
    pub async fn new(canvas_id: &str) -> Result<WasmViewer, JsValue> {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).ok();

        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?
            .dyn_into::<HtmlCanvasElement>()?;

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| e.to_string())?;
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("No GPU adapter")?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        }, None).await.map_err(|e| e.to_string())?;

        let width = canvas.width();
        let height = canvas.height();
        let config = surface.get_default_config(&adapter, width, height)
            .ok_or("Surface config failed")?;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let bgl_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl_compute = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute PL"), bind_group_layouts: &[&bgl_compute], push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"), layout: Some(&pl_compute), module: &shader, entry_point: Some("compute_main"),
            compilation_options: Default::default(), cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform"), size: 128, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let bgl_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl_render = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render PL"), bind_group_layouts: &[&bgl_render], push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pl_render),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 48, // 16*3 bytes (Surfel)
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },  // pos
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 1 }, // color
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 32, shader_location: 2 }, // normal
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::PointList, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        let bg_render = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render BG"), layout: &bgl_render, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }
            ],
        }));

        let camera = Rc::new(RefCell::new(CameraController::new(width as f32, height as f32)));
        let mut closures = Vec::new();

        {
            let cam = camera.clone();
            let is_dragging = Rc::new(RefCell::new(false));
            let last_pos = Rc::new(RefCell::new((0.0, 0.0)));
            let drag_mode = Rc::new(RefCell::new(0));

            let c_drag = is_dragging.clone();
            let c_pos = last_pos.clone();
            let c_mode = drag_mode.clone();
            let closure = Closure::<dyn FnMut(_)>::new(move |e: MouseEvent| {
                *c_drag.borrow_mut() = true;
                *c_pos.borrow_mut() = (e.offset_x() as f32, e.offset_y() as f32);
                if e.button() == 2 { *c_mode.borrow_mut() = 2; } else { *c_mode.borrow_mut() = 1; }
            });
            canvas.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref()).unwrap();
            closures.push(closure.into_js_value());

            let c_drag = is_dragging.clone();
            let closure = Closure::<dyn FnMut(_)>::new(move |_: MouseEvent| { *c_drag.borrow_mut() = false; });
            window.add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref()).unwrap();
            closures.push(closure.into_js_value());

            let closure = Closure::<dyn FnMut(_)>::new(move |e: MouseEvent| { e.prevent_default(); });
            canvas.add_event_listener_with_callback("contextmenu", closure.as_ref().unchecked_ref()).unwrap();
            closures.push(closure.into_js_value());

            let c_drag = is_dragging.clone();
            let c_pos = last_pos.clone();
            let c_mode = drag_mode.clone();
            let c_cam = cam.clone();
            let closure = Closure::<dyn FnMut(_)>::new(move |e: MouseEvent| {
                if *c_drag.borrow() {
                    let cx = e.offset_x() as f32;
                    let cy = e.offset_y() as f32;
                    let (lx, ly) = *c_pos.borrow();
                    if *c_mode.borrow() == 1 { c_cam.borrow_mut().rotate(cx - lx, cy - ly); }
                    else if *c_mode.borrow() == 2 { c_cam.borrow_mut().pan(cx - lx, cy - ly); }
                    *c_pos.borrow_mut() = (cx, cy);
                }
            });
            canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref()).unwrap();
            closures.push(closure.into_js_value());

            let c_cam = cam.clone();
            let closure = Closure::<dyn FnMut(_)>::new(move |e: WheelEvent| {
                e.prevent_default();
                c_cam.borrow_mut().zoom(e.delta_y() as f32);
            });
            canvas.add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref()).unwrap();
            closures.push(closure.into_js_value());
        }

        Ok(Self {
            device, queue, surface, config, render_pipeline, compute_pipeline,
            bg_compute: None, bg_render, bgl_compute, bgl_render,
            uniform_buffer, vertex_buffer: None, num_vertices: 0,
            camera, display_mode: 0, _closures: closures,
        })
    }

    pub fn load_data(&mut self, data: &[u8]) {
        let header_end = b"end_header";
        let offset = data.windows(header_end.len()).position(|w| w == header_end).map(|i| i + header_end.len());
        if let Some(mut cursor) = offset {
            while cursor < data.len() && (data[cursor] == b'\r' || data[cursor] == b'\n') { cursor += 1; }
            let raw_data = &data[cursor..];
            let struct_size = std::mem::size_of::<RawSplat>();
            let count = raw_data.len() / struct_size;
            if count == 0 { return; }
            
            let raw_splats: &[RawSplat] = bytemuck::cast_slice(&raw_data[..count*struct_size]);
            let mut splats = Vec::with_capacity(count);
            for raw in raw_splats {
                splats.push(GaussianSplat {
                    pos: [raw.x, raw.y, raw.z], opacity: raw.opacity,
                    scale: [raw.scale_0, raw.scale_1, raw.scale_2], _pad1: 0.0,
                    rot: [raw.rot_0, raw.rot_1, raw.rot_2, raw.rot_3],
                    sh_dc: [raw.f_dc_0, raw.f_dc_1, raw.f_dc_2], _pad2: 0.0,
                });
            }

            let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input"), contents: bytemuck::cast_slice(&splats), usage: wgpu::BufferUsages::STORAGE,
            });
            let output_size = (count * std::mem::size_of::<Surfel>()) as u64;
            let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output"), size: output_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX, mapped_at_creation: false,
            });

            let bg_compute = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute BG"), layout: &self.bgl_compute,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.compute_pipeline);
                pass.set_bind_group(0, &bg_compute, &[]);
                pass.dispatch_workgroups((count as u32 + 63) / 64, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));

            self.vertex_buffer = Some(output_buf);
            self.num_vertices = count as u32;
            self.bg_compute = Some(bg_compute);
            log::info!("Loaded {} splats.", count);
        }
    }

    pub fn set_display_mode(&mut self, mode: u32) { self.display_mode = mode; }
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width; self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.camera.borrow_mut().update_resolution(width as f32, height as f32);
        }
    }
    pub fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex, Err(_) => return,
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = self.camera.borrow().get_uniform(self.display_mode);
        self.queue.write_buffer(&self.uniform_buffer, 0, &uniform);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            if let Some(vb) = &self.vertex_buffer {
                pass.set_pipeline(&self.render_pipeline);
                if let Some(bg) = &self.bg_render { pass.set_bind_group(0, bg, &[]); }
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..self.num_vertices, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
    }
}
