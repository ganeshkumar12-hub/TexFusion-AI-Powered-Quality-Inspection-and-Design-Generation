What is TexFusion?
TexFusion is a comprehensive deep-learning system built to modernize textile manufacturing. It analyzes fabric images for defects using CNNs, identifies pattern categories using EfficientNetB3, and generates new textile designs using a conditional DCGAN model. The platform provides fast, accurate, and creative support for both quality assurance and textile design teams.
Features
Automated Fabric Defect Detection
Detects stains, holes, weave faults, horizontal/vertical line defects using a CNN-based inspection model.
Provides confidence scores and top-3 predictions for improved quality assessment.
Ensures consistent and reliable defect detection compared to manual inspection.

Pattern Recognition (19 Textile Categories)
EfficientNetB3 architecture trained on 19 fabric pattern classes for high-accuracy classification.
Supports recognition of patterns such as floral, stripes, geometric, checks, abstract, and more.
Helps designers and manufacturers categorize fabrics for inventory and production workflows.

AI Design Generation (Conditional DCGAN)
Generates new textile patterns conditioned on selected pattern categories.
Produces unique, visually coherent, and creative fabric designs using PyTorch-based GAN models.
Allows rapid prototyping of designs without manual drawing.

Color & Style Customization
Real-time hue, saturation, and brightness adjustments on generated designs.
Applies enhancement filters such as sharpen, smooth, and stylize for aesthetic refinement.
Enables quick experimentation with multiple color variations.

Motif Overlay & Tiling
Upload custom motifs (PNG) and overlay them on generated backgrounds.
Supports repeating motifs in tiled or centered formats, ideal for fabric print layouts.
Creates production-ready textile patterns with both background and motif layers.

Interactive Web Interface
Simple upload-based workflow for inspection, pattern recognition, and design generation.
Displays predictions, confidence levels, and design previews in real time.
Allows users to download final textile designs instantly.

Optimized Backend Architecture
Flask backend integrating CNN, EfficientNetB3, and DCGAN models through dedicated REST APIs.
Seamless communication with the frontend for fast inference and design generation.
Robust processing pipeline with image preprocessing, prediction, post-processing, and rendering.
Why TexFusion?
Improved Fabric Quality Control: Automates defect detection using AI, reducing manual errors and ensuring consistent inspection standards across textile batches.
Fast & Accurate Pattern Identification: EfficientNetB3-based pattern recognition helps categorize fabrics instantly, supporting designers, manufacturers, and inventory teams.
AI-Powered Design Innovation: The conditional DCGAN model enables rapid generation of new textile patterns, helping designers explore creative ideas without manual sketching.
End-to-End Textile Workflow: Combines quality inspection, pattern recognition, and design generation into one unified platform for maximum productivity.
Customization-Focused Tools: Offers dynamic color tweaking, enhancement filters, and motif overlays, allowing users to create production-ready designs tailored to their needs.
Fast, Lightweight, and User-Friendly: The web-based interface provides an intuitive workflow with real-time previews, easy image uploads, and instant design downloads.
