use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::{GaussianSplat, Surfel};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SrParams {
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct SuperResolutionPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SuperResolutionPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SR Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sr.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SR Bind Group Layout"),
            entries: &[
                // Input Splats
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // Output Surfels
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SR Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SR Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_sr"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout }
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_splats: &[GaussianSplat],
        factor: u32,
    ) -> (wgpu::Buffer, u32) {
        let input_count = input_splats.len() as u32;
        let output_count = input_count * factor;
        let output_size = (output_count as usize * std::mem::size_of::<Surfel>()) as u64;

        // 1. Create Buffers
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SR Input Buffer"),
            contents: bytemuck::cast_slice(input_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffer acts as Vertex Buffer for rendering or Staging for reading
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SR Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SrParams { factor, _pad0: 0, _pad1: 0, _pad2: 0 };
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SR Param Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SR Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
            ],
        });

        // 3. Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("SR Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (output_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        (output_buffer, output_count)
    }

    // Python用: 結果をCPUに読み戻すヘルパー
    pub async fn run_and_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_splats: &[GaussianSplat],
        factor: u32,
    ) -> Result<Vec<Surfel>, String> {
        let (output_buffer, output_count) = self.run(device, queue, input_splats, factor);
        let output_size = (output_count as usize * std::mem::size_of::<Surfel>()) as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

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
}
