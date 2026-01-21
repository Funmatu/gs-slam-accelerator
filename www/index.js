import init, { WasmViewer } from './pkg/gs_slam_core.js';

async function run() {
    await init();
    console.log("WASM Initialized");

    const statusDiv = document.getElementById('status');
    const canvas = document.getElementById('canvas');
    const btnRGB = document.getElementById('btnRGB');
    const btnNormal = document.getElementById('btnNormal');
    const btnExport = document.getElementById('btnExport');

    // Resize Handling
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        if (window.viewer) window.viewer.resize(canvas.width, canvas.height);
    }
    window.addEventListener('resize', resize);
    resize();

    try {
        const viewer = await WasmViewer.new("canvas");
        window.viewer = viewer; // Make global for resize event
        statusDiv.innerText = "Ready.";

        // Animation Loop
        function animate() {
            viewer.render();
            requestAnimationFrame(animate);
        }
        animate();

        // File Loading
        document.getElementById('fileInput').addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            statusDiv.innerText = `Loading ${file.name}...`;
            btnExport.disabled = true;
            try {
                const buffer = await file.arrayBuffer();
                const data = new Uint8Array(buffer);
                console.log(`Passing ${data.length} bytes to WASM`);
                viewer.load_data(data);
                statusDiv.innerText = `Rendering ${file.name} (${(data.length/1024/1024).toFixed(1)} MB)`;
                btnExport.disabled = false;
            } catch (err) {
                console.error(err);
                statusDiv.innerText = "Error loading file";
            }
        });

        // Mode Switching
        btnRGB.onclick = () => {
            viewer.set_display_mode(0);
            btnRGB.classList.add('active');
            btnNormal.classList.remove('active');
        };
        btnNormal.onclick = () => {
            viewer.set_display_mode(1);
            btnRGB.classList.remove('active');
            btnNormal.classList.add('active');
        };

        // Export Functionality
        btnExport.onclick = () => {
            if (!window.viewer) return;
            statusDiv.innerText = "Generating PLY...";
            
            // Allow UI update before freezing for generation
            setTimeout(() => {
                try {
                    const data = window.viewer.export_ply();
                    const blob = new Blob([data], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);
                    
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = "processed_geometry.ply";
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    statusDiv.innerText = "Export Complete.";
                } catch (e) {
                    console.error(e);
                    statusDiv.innerText = "Export Failed.";
                }
            }, 10);
        };

    } catch (e) {
        console.error("Initialization failed:", e);
        statusDiv.innerText = "WebGPU initialization failed. Check console.";
    }
}

run();
