# TexFusion – AI-Powered Textile Defect Detection, Pattern Recognition & Design Generation
TexFusion is an AI-powered textile automation platform that integrates fabric defect detection, pattern recognition, and GAN-based design generation. It helps textile industries improve quality control while enabling rapid creation of new fabric patterns. The system provides a unified, intelligent workflow combining computer vision and generative AI.
<h2>Table of Contents</h2>
<ul>
  <li> <a href = "#About"> About </a></li>
  <ul>
   <li><a href="#wa"> What is TexFusion? </a></li> 
   <li><a href="#features"> Features </a></li> 
   <li><a href="#why"> Why TexFusion? </a></li>
  </ul>
  <li> <a href = "#getting_started"> Getting Started </a></li>
  <ul>
   <li><a href="#prerequisites"> Prerequisites </a></li> 
   <li><a href="#installation"> Installation </a></li> 
   <li><a href="#backend_setup"> Backend Setup </a></li>
   <li><a href="#frontend_setup"> Building the App </a></li>
  </ul>
  <li> <a href = "#tech_used"> TechStack Used </a></li>
  <li> <a href = "#architecture"> System Architecture </a></li>
  <li> <a href = "#screenshots"> Screenshots and App Demonstration </a></li>
  <li> <a href = "#conclusion"> Conclusion </a></li>
  <li> <a href = "#team"> Developed By </a></li>
</ul>
<section id = "About">
  <h2> About </h2>
  <h3 id = "wa"> What is TexFusion? </h3>
    TexFusion is a comprehensive deep-learning system built to modernize textile manufacturing. It analyzes fabric images for defects using CNNs, identifies pattern categories using EfficientNetB3, and generates new textile designs using a conditional DCGAN model. The platform provides fast, accurate, and creative support for both quality assurance and textile design teams.
  <h3 id = "features"> Features </h3>
<ul>
    <li><strong>Automated Fabric Defect Detection</strong>
        <ul>
            <li>Detects stains, holes, weave faults, horizontal/vertical line defects using a CNN-based inspection model.</li>
            <li>Provides confidence scores and top-3 predictions for improved quality assessment.</li>
            <li>Ensures consistent and reliable defect detection compared to manual inspection.</li>
        </ul>
    </li>
    <br>
    <li><strong>Pattern Recognition (19 Textile Categories)</strong>
        <ul>
            <li>EfficientNetB3 architecture trained on 19 fabric pattern classes for high-accuracy classification.</li>
            <li>Supports recognition of patterns such as floral, stripes, geometric, checks, abstract, and more.</li>
            <li>Helps designers and manufacturers categorize fabrics for inventory and production workflows.</li>
        </ul>
    </li>
    <br>
    <li><strong>AI Design Generation (Conditional DCGAN)</strong>
        <ul>
            <li>Generates new textile patterns conditioned on selected pattern categories.</li>
            <li>Produces unique, visually coherent, and creative fabric designs using PyTorch-based GAN models.</li>
            <li>Allows rapid prototyping of designs without manual drawing.</li>
        </ul>
    </li>
    <br>
    <li><strong>Color & Style Customization</strong>
        <ul>
            <li>Real-time hue, saturation, and brightness adjustments on generated designs.</li>
            <li>Applies enhancement filters such as sharpen, smooth, and stylize for aesthetic refinement.</li>
            <li>Enables quick experimentation with multiple color variations.</li>
        </ul>
    </li>
    <br>
    <li><strong>Motif Overlay & Tiling</strong>
        <ul>
            <li>Upload custom motifs (PNG) and overlay them on generated backgrounds.</li>
            <li>Supports repeating motifs in tiled or centered formats, ideal for fabric print layouts.</li>
            <li>Creates production-ready textile patterns with both background and motif layers.</li>
        </ul>
    </li>
    <br>
    <li><strong>Interactive Web Interface</strong>
        <ul>
            <li>Simple upload-based workflow for inspection, pattern recognition, and design generation.</li>
            <li>Displays predictions, confidence levels, and design previews in real time.</li>
            <li>Allows users to download final textile designs instantly.</li>
        </ul>
    </li>
    <br>
    <li><strong>Optimized Backend Architecture</strong>
        <ul>
            <li>Flask backend integrating CNN, EfficientNetB3, and DCGAN models through dedicated REST APIs.</li>
            <li>Seamless communication with the frontend for fast inference and design generation.</li>
            <li>Robust processing pipeline with image preprocessing, prediction, post-processing, and rendering.</li>
        </ul>
    </li>
</ul>  
 <h3 id="why"> Why TexFusion? </h3>
<ul>
    <li><strong>Improved Fabric Quality Control</strong>: Automates defect detection using AI, reducing manual errors and ensuring consistent inspection standards across textile batches.</li>
    <li><strong>Fast & Accurate Pattern Identification</strong>: EfficientNetB3-based pattern recognition helps categorize fabrics instantly, supporting designers, manufacturers, and inventory teams.</li>
    <li><strong>AI-Powered Design Innovation</strong>: The conditional DCGAN model enables rapid generation of new textile patterns, helping designers explore creative ideas without manual sketching.</li>
    <li><strong>End-to-End Textile Workflow</strong>: Combines quality inspection, pattern recognition, and design generation into one unified platform for maximum productivity.</li>
    <li><strong>Customization-Focused Tools</strong>: Offers dynamic color tweaking, enhancement filters, and motif overlays, allowing users to create production-ready designs tailored to their needs.</li>
    <li><strong>Fast, Lightweight, and User-Friendly</strong>: The web-based interface provides an intuitive workflow with real-time previews, easy image uploads, and instant design downloads.</li>
</ul>
</section>
<section id="getting_started">
  <h2> Getting Started </h2>

  <h3 id="prerequisites"> Prerequisites </h3>
  <p>Before you begin, ensure that the following software and dependencies are installed in your development environment:</p>

  <h4>For Backend (Flask + AI Models):</h4>
  <ul>
    <li>
      <strong>Python 3.8+</strong>: Required for running the TexFusion backend and all deep-learning models.
      <ul>
        <li><a href="https://www.python.org/downloads/">Download Python</a></li>
      </ul>
    </li>
    <li><strong>pip</strong>: Python package manager used to install project dependencies.</li>
    <li>
      <strong>Required Python Libraries</strong>:
      <ul>
        <li>TensorFlow / Keras (for CNN & EfficientNetB3 models)</li>
        <li>PyTorch (for Conditional DCGAN)</li>
        <li>OpenCV (for image processing & color adjustments)</li>
        <li>NumPy, Pillow, Flask, Flask-CORS</li>
      </ul>
    </li>
    <li>
      <strong>Pretrained Model Files</strong>:
      <ul>
        <li>CNN model for fabric defect detection</li>
        <li>EfficientNetB3 pattern recognition model</li>
        <li>DCGAN generator weights for textile design generation</li>
      </ul>
    </li>
  </ul>

  <h4>For Frontend (Web Interface):</h4>
  <ul>
    <li><strong>Any modern web browser</strong> (Chrome, Edge, Firefox).</li>
    <li><strong>Basic static server</strong> (optional): You can simply open <code>index.html</code> directly or use VS Code Live Server.</li>
    <li>Ensure backend API URLs are updated inside your JavaScript files.</li>
  </ul>

  <h3 id="installation"> Installation </h3>

  <h4>Clone the Repository:</h4>
  <pre><code>git clone https://github.com/Suhas-Varna/TexFusion.git
cd TexFusion</code></pre>

  <h3 id="backend_setup"> Backend Setup (Flask) </h3>
  <ol>
    <li>
      <p><strong>Create Virtual Environment</strong> (Recommended):</p>
      <pre><code>python -m venv venv</code></pre>
    </li>
    <li>
      <p><strong>Activate Virtual Environment</strong>:</p>
      <pre><code># Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate</code></pre>
    </li>
    <li>
      <p><strong>Start the Flask Backend Server</strong>:</p>
      <pre><code>python app.py</code></pre>
      <p>The backend will run at: <code>http://127.0.0.1:5000/</code></p>
    </li>
    <li>
      <p><strong>Ensure Model Paths Are Correct</strong>:</p>
      <p>Update the paths for: CNN model, EfficientNetB3 model, and DCGAN weights inside <code>app.py</code>.</p>
    </li>
  </ol>
  <h3 id="frontend_setup"> Frontend Setup </h3>
  <ol>
    <li><strong>Navigate to the frontend directory</strong> (if applicable) or simply open the <code>index.html</code> file.</li>
    <li>
      <p><strong>Update API Base URL</strong> inside your JavaScript:</p>
      <pre><code>const BASE_URL = "http://127.0.0.1:5000";</code></pre>
    </li>
    <li>
      <p><strong>Run the frontend</strong>:</p>
      <ul>
        <li>Option 1: Open <code>index.html</code> directly in browser</li>
        <li>Option 2: Use VS Code → Live Server</li>
      </ul>
    </li>
  </ol>
</section>


<section id="tech_used">
  <h2> TechStack - Built With
    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968350.png" alt="Python" height="20" style="vertical-align: middle;"/>
    <img src="https://github.com/user-attachments/assets/3ce45ba2-daf6-4938-aff9-87e8f7063ac5" alt="Flask" height="20" style="vertical-align: middle;"/>
    <img src="https://github.com/user-attachments/assets/92b0557a-ed84-4fbc-92ce-acb925715986" alt="TensorFlow" height="20" style="vertical-align: middle;"/>
    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968267.png" alt="JS" height="20" style="vertical-align: middle;"/>
  </h2>

  <p><strong>Python:</strong> Core programming language used for building the backend and all AI model pipelines (CNN, EfficientNetB3, DCGAN).</p>

  <p><strong>Flask:</strong> Lightweight web framework used to serve the three TexFusion APIs — defect detection, pattern recognition, and design generation.</p>

  <p><strong>TensorFlow/Keras:</strong> Used for training and deploying the CNN defect detection model and EfficientNetB3 pattern recognition classifier.</p>

  <p><strong>PyTorch:</strong> Framework used to build and run the Conditional DCGAN responsible for textile design generation.</p>

  <p><strong>OpenCV:</strong> Handles image preprocessing, HSV color adjustments, enhancement filters, and motif overlay operations.</p>

  <p><strong>HTML, CSS, JavaScript:</strong> Used to create a simple yet interactive web interface that allows users to upload images, preview outputs, and download generated designs.</p>
</section>

<section id="architecture">
  <h2> System Architecture </h2>

  <h3>🏗️ High-Level Architecture</h3>

<pre>
┌─────────────────────────────────────────────────────────────────────────────┐
│                                TexFusion APP                                │
│   ┌───────────────────┐   ┌─────────────────────┐   ┌────────────────────┐  │
│   │   Home Screen     │ → │ Image Upload Module │ → │  API Request Layer │  │
│   └───────────────────┘   └─────────────────────┘   └────────────────────┘  │
│                ↓                       ↓                       ↓            │
│      ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│      │ Defect Detection  │   │ Pattern Classifier│   │ Design Generator  │  │
│      │      (CNN)        │   │ (EfficientNetB3)  │   │  (DCGAN Model)    │  │
│      └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│                ↓                       ↓                       ↓            │
│     ┌────────────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
│     │ JSON Predictions   │   │ Pattern Labels     │   │ Generated Images │  │
│     └────────────────────┘   └────────────────────┘   └──────────────────┘  │
└──────────────────────────────────────│──────────────────────────────────────┘
                                       │ API CALLS / JSON RESPONSE
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │          Request Router & Processing Engine                          │  │
│   │  • Routes user-uploaded images to correct model                      │  │
│   │  • Handles CNN, EfficientNet, and GAN inference                      │  │
│   │  • Returns predictions or generated designs                          │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│     │                       │                           │                   │
│     ▼                       ▼                           ▼                   │
│  /detect-defect       /classify-pattern             /generate-design        │
└─────────────────────────────────────────────────────────────────────────────┘
</pre>

  <h3>📊 Data Flow Diagram</h3>

<pre>
  USER UPLOADS FABRIC IMAGE
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                TexFusion Frontend                   │
│  • Uploads image                                    │
│  • Selects feature: Defect / Pattern / Design       │
└───────────────────┬─────────────────────────────────┘
                    │  HTTP POST (multipart image)
                    ▼
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                     │
│  • Accepts image                                    │
│  • Validates and preprocesses                       │
│  • Forwards to respective ML module                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│                 MODEL PROCESSING                    │
│  • CNN → Detects six defect classes                 │
│  • EfficientNetB3 → Predicts 19 textile patterns    │
│  • Conditional DCGAN → Generates new designs        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│                 JSON Response / Image Output        │
│  • Predicted class + confidence                     │
│  • Suggested pattern group                          │
│  • Generated textile design image                   │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│                 TexFusion Frontend                  │
│  • Displays results                                 │
│  • Allows color edits & motif overlays (GAN)        │
│  • Supports download of final design                │
└─────────────────────────────────────────────────────┘
</pre>

  <h3>🗂️ Project Structure</h3>

<pre>
TexFusion/
│
├── frontend/                       # Web UI (HTML, CSS, JS)
│   ├── index.html
│   ├── upload.css
│   ├── design.js
│
├── backend/                        # FastAPI Server
│   ├── main.py                     # API endpoints
│   ├── defect_model/               # CNN model files
│   ├── pattern_model/              # EfficientNetB3 model files
│   ├── gan_model/                  # DCGAN generator + embeddings
│   ├── utils/                      # Preprocessing, helpers
│   └── requirements.txt
│
└── README.md
</pre>
<p>Note: Due to large file sizes, the trained model files are excluded from this repository. Please contact us via email to obtain access to the models.</p>

  <h3>🔐 TexFusion Security Architecture</h3>
<ul>
  <li><strong>Secure Model Access</strong>:
    <ul>
      <li>All ML models run locally on the server</li>
      <li>No external API dependency</li>
      <li>No cloud upload of user data</li>
    </ul>
  </li>

  <li><strong>Data Privacy</strong>:
    <ul>
      <li>Uploaded images processed in-memory only</li>
      <li>No images or metadata stored on server</li>
      <li>Automatic cleanup of temp files</li>
    </ul>
  </li>

  <li><strong>API Security</strong>:
    <ul>
      <li>CORS restricted to trusted UI origins</li>
      <li>Validation on image size, format, and request type</li>
      <li>Rate-limiting for design generation requests</li>
    </ul>
  </li>
</ul>

  <h3>⚡ TexFusion Performance Optimizations</h3>
<ul>
  <li><strong>Backend Optimizations</strong>:
    <ul>
      <li>Efficient batch preprocessing</li>
      <li>Model warm-loading for faster inference</li>
      <li>GPU-accelerated GAN generation (optional)</li>
    </ul>
  </li>

  <li><strong>Frontend Optimizations</strong>:
    <ul>
      <li>Lazy-loaded image previews</li>
      <li>Client-side color filters using Canvas API</li>
      <li>Compressed API responses for faster rendering</li>
    </ul>
  </li>
</ul>
</section>

<section id="screenshots">
  <h2 id="screenshots">App Demonstration</h2>  
  <h2> Screenshots </h2>   
  <img src="https://github.com/user-attachments/assets/f126d77e-7b86-4fbc-8ed7-fff08ea57b81" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/26330225-bca8-4620-a36b-43a3146fc6d2" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/6c3e8607-2217-4f36-a252-067829d6eb84" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/0773e21f-215a-40f2-8eed-22c82699927f" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/18c8aa61-9c29-48f1-894e-b3f735d0823f" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/399e6604-801d-44f1-bb05-4c14cb206109" style="width: 200px;" />
  <img src="https://github.com/user-attachments/assets/2f63bcbd-b6a0-4285-85bd-4ed405a0b681" style="width: 200px;" />
</section>


<section id="conclusion">
  <h2>Conclusion</h2>
  <p>
   TexFusion successfully integrates defect detection, pattern recognition, and AI-driven design generation into a unified textile intelligence platform. By combining CNNs, EfficientNetB3, and a Conditional DCGAN, the system automates critical manufacturing and creative processes with high reliability. Its interactive web interface enables real-time inspection and customizable design generation, reducing manual effort and streamlining workflows. Overall, TexFusion demonstrates how AI can enhance productivity, accuracy, and innovation in the textile industry.
  </p>
</section>



<section id = "team">
  <h2> The Team </h2>
  <h3> Suhas Varna </h3>
<p align="left">
  <a href="https://github.com/Suhas-Varna" style="text-decoration: none;" target="_blank" rel="nofollow">
    <img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github" alt="GitHub" style="max-width: 100%;">
  </a>
  <a href="https://www.linkedin.com/in/suhas-varna2003/" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin" alt="LinkedIn" />
  </a>
</p>

<h3> Seeripi Ganesh Kumar  </h3>
<p align="left">
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github" alt="GitHub" />
  </a>
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin" alt="LinkedIn" />
  </a>
</p>

<h3> Vikas D H </h3>
<p align="left">
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github" alt="GitHub" />
  </a>
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin" alt="LinkedIn" />
  </a>
</p>

<h3> Sanjay J </h3>
<p align="left">
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github" alt="GitHub" />
  </a>
  <a href="" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin" alt="LinkedIn" />
  </a>
</p>
</section>


