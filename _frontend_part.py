

# ════════════════════════════════════════════════════════════
# GRADIO APP — Native Gradio + DotField Background
# ════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

body {
    font-family: 'Inter', sans-serif !important;
    background: #120F17 !important;
    color: #e6edf3 !important;
}
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    color: #e6edf3 !important;
    position: relative;
    z-index: 5;
}
footer { display: none !important; }

/* DotField canvas */
#dotfield-bg {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none;
}

/* Hero */
.hero-wrap {
    text-align: center;
    padding: 40px 20px 28px;
    position: relative; z-index: 5;
}
.hero-wrap h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a855f7, #c084fc, #e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px !important;
}
.hero-wrap p {
    color: #9ca3af !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
}
.pill-row {
    display: flex; justify-content: center; gap: 10px;
    flex-wrap: wrap; margin-top: 18px;
}
.pill-item {
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.2);
    border-radius: 50px; padding: 5px 14px;
    font-size: 0.75rem; color: #c4b5fd;
}
.pill-item strong { color: #a855f7; }

/* Tabs */
.tabs > .tab-nav {
    background: rgba(168,85,247,0.06) !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-bottom: none !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 6px 6px 0 !important;
    backdrop-filter: blur(10px);
}
.tabs > .tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 22px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.tabs > .tab-nav button:hover {
    color: #c084fc !important;
    background: rgba(168,85,247,0.08) !important;
}
.tabs > .tab-nav button.selected {
    color: #c084fc !important;
    background: rgba(168,85,247,0.13) !important;
    border-bottom: 2px solid #a855f7 !important;
}

/* Tab panels */
.tabitem {
    background: rgba(18,15,23,0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(168,85,247,0.12) !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    padding: 28px !important;
}

/* Inputs */
textarea, input[type=text] {
    background: rgba(18,15,23,0.85) !important;
    color: #d1d5db !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(168,85,247,0.4) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(168,85,247,0.15) !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #7c3aed, #a855f7, #c026d3) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(168,85,247,0.35) !important;
    font-family: 'Inter', sans-serif !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(168,85,247,0.5) !important;
}

/* Labels */
label span {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Section desc */
.sec-desc {
    font-size: 0.82rem !important;
    color: #6b7280 !important;
    margin-bottom: 16px !important;
}

/* Footer */
.app-footer {
    text-align: center; margin-top: 28px; padding-top: 18px;
    border-top: 1px solid rgba(168,85,247,0.08);
    color: #374151; font-size: 0.74rem;
}
.app-footer a { color: #7c3aed; text-decoration: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.25); border-radius: 3px; }
"""

DOTFIELD_JS = """
<canvas id="dotfield-bg"></canvas>
<script>
(function(){
  var cv=document.getElementById('dotfield-bg');
  if(!cv){setTimeout(arguments.callee,200);return;}
  var ctx=cv.getContext('2d');
  var R=1.5,SP=14,CR=600,CF=0.1,BS=67,GR=160;
  var mx=-9999,my=-9999,dots=[];
  function resize(){
    cv.width=innerWidth;cv.height=innerHeight;dots=[];
    var cols=Math.ceil(cv.width/SP)+2,rows=Math.ceil(cv.height/SP)+2;
    for(var r=0;r<=rows;r++)for(var c=0;c<=cols;c++)dots.push({ox:c*SP,oy:r*SP});
  }
  function clr(d){
    if(d>=CR)return'rgba(180,151,207,0.07)';
    var t=d/CR,
      rv=Math.round(168+(180-168)*t),
      gv=Math.round(85+(151-85)*t),
      bv=Math.round(247+(207-247)*t),
      glow=Math.max(0,1-d/GR),
      a=Math.min(1,(0.35+(0.25-0.35)*t)+glow*0.65);
    return'rgba('+rv+','+gv+','+bv+','+a.toFixed(3)+')';
  }
  function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    for(var i=0;i<dots.length;i++){
      var dot=dots[i],dx=dot.ox-mx,dy=dot.oy-my,
        dist=Math.sqrt(dx*dx+dy*dy),x=dot.ox,y=dot.oy;
      if(dist<CR&&dist>0.01){
        var ratio=1-dist/CR,disp=ratio*ratio*BS*CF,
          ang=Math.atan2(dy,dx);
        x=dot.ox+Math.cos(ang)*disp;
        y=dot.oy+Math.sin(ang)*disp;
      }
      ctx.beginPath();ctx.arc(x,y,R,0,Math.PI*2);
      ctx.fillStyle=clr(dist);ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;});
  addEventListener('mouseleave',function(){mx=-9999;my=-9999;});
  addEventListener('resize',resize);
  resize();draw();
})();
</script>
"""

BIRD_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#120F17",
    body_background_fill_dark="#120F17",
    block_background_fill="rgba(18,15,23,0.5)",
    block_background_fill_dark="rgba(18,15,23,0.5)",
    block_border_color="rgba(168,85,247,0.1)",
    block_border_color_dark="rgba(168,85,247,0.1)",
    block_label_text_color="#9ca3af",
    block_label_text_color_dark="#9ca3af",
    input_background_fill="rgba(18,15,23,0.85)",
    input_background_fill_dark="rgba(18,15,23,0.85)",
    button_primary_background_fill="linear-gradient(135deg,#7c3aed,#a855f7)",
    button_primary_background_fill_dark="linear-gradient(135deg,#7c3aed,#a855f7)",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(title="Bird Species Identifier") as app:

    # ── DotField background ──
    gr.HTML(DOTFIELD_JS)

    # ── Hero ──
    gr.HTML("""
    <div class="hero-wrap">
      <h1>🐦 Bird Species Identifier</h1>
      <p>AI-powered recognition · EfficientNetV2-S · 200 species</p>
      <div class="pill-row">
        <span class="pill-item">🧠 Model <strong>EfficientNetV2-S</strong></span>
        <span class="pill-item">🦜 Species <strong>200</strong></span>
        <span class="pill-item">📷 Input <strong>380×380 px</strong></span>
        <span class="pill-item">⚡ Device <strong>CPU / CUDA</strong></span>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Identify Bird ──
        with gr.Tab("🔍  Identify Bird"):
            gr.HTML('<p class="sec-desc">Upload a bird photo — get species, habitat, migration info and top 5 predictions</p>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Bird Photo", type="numpy", height=340)
                    identify_btn = gr.Button("🔍  Identify Species", variant="primary")
                with gr.Column(scale=1):
                    txt_output = gr.Textbox(label="Identification Results", lines=22,
                                            placeholder="Results will appear here…")
            identify_btn.click(fn=predict_bird, inputs=img_input, outputs=txt_output)

        # ── Tab 2: Grad-CAM ──
        with gr.Tab("🔥  Grad-CAM"):
            gr.HTML("""<p class="sec-desc">See what part of the image the model focuses on ·
              <span style="color:#f97316;">Red/Yellow</span> = high attention ·
              <span style="color:#60a5fa;">Blue</span> = low attention</p>""")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    cam_input = gr.Image(label="Upload Bird Photo", type="numpy", height=340)
                    cam_btn = gr.Button("🔥  Generate Grad-CAM Heatmap", variant="primary")
                with gr.Column(scale=1):
                    cam_output = gr.Image(label="Grad-CAM Heatmap", height=300)
                    cam_text = gr.Textbox(label="Analysis Result", lines=5,
                                          placeholder="Heatmap result will appear here…")
            cam_btn.click(fn=generate_gradcam, inputs=cam_input, outputs=[cam_output, cam_text])

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
      🐦 Bird Species Identifier · EfficientNetV2-S · CUB-200-2011 ·
      Built with <a href="https://gradio.app" target="_blank">Gradio</a>
    </div>
    """)

print("\n✅ Starting Bird Identifier App...")
print("   Open in browser: http://127.0.0.1:7860\n")
app.launch(theme=BIRD_THEME, css=CUSTOM_CSS)
