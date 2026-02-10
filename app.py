# ============================================
# Lung Disease Detection App (Streamlit)
# Fully commented version (detailed) - READY for viva/presentation
# Team (in-progress responsibilities shown next to sections)
# - Md Kaif   : (Lead) Model & App integration, Grad-CAM integration  [Tech: PyTorch, Torchvision, Grad-CAM, Streamlit]
# - Md Zuhaib : Dataset & Preprocessing                             [Tech: OpenCV, PIL, Torchvision, NumPy]
# - Mohammad Adil: Local Chatbot Integration                        [Tech: HuggingFace/DialoGPT, PyTorch, Streamlit]
# - Mohammad Shahil: Documentation, UI testing, PDF report          [Tech: FPDF, Streamlit]
# ============================================

# ==== Required Libraries ====
# (All standard libs used across modules: UI, DL, image processing, reporting, chatbot)
import os                         # (general file operations if needed)
import streamlit as st            # (UI framework)  <-- Main app interface (Kaif + Shahil)
import torch                      # (PyTorch: model inference & tensors)  <-- Core ML (Kaif)
import torch.nn as nn             # (Neural network building blocks)
from torchvision import transforms, models  # (Transforms & model architectures)  <-- Preprocessing & ResNet (Zuhaib + Kaif)
from PIL import Image             # (Pillow: image loading & conversion)  <-- Image IO (Zuhaib)
from datetime import datetime    # (for timestamp in PDF)  <-- Report (Shahil)
from fpdf import FPDF            # (Generate PDF reports programmatically)  <-- Report (Shahil)
import numpy as np               # (Numerical ops)  <-- Preprocessing & Grad-CAM (Zuhaib/Kaif)
import cv2                       # (OpenCV: color mapping, overlays, resizing)  <-- Visualization (Zuhaib/Kaif)
from chatbot_module import LocalChatbot  # 👈 Local chatbot module (Adil)
# Note: chatbot_module is assumed to be implemented separately (DialoGPT wrapper). (Adil)

# ==== Streamlit Config ====
# (UI page title and layout configuration)  <-- UI (Shahil + Kaif)
st.set_page_config(page_title="Lung Detector + AI Chat", layout="wide")

# ==== Initialize Session State ====
# Streamlit re-runs the script top-to-bottom on interactions.
# Use st.session_state to persist variables (patient data, prediction) across reruns.
# (Owner: Kaif + Shahil)
for key in ["name", "age", "gender", "symptoms", "image", "pred_class",
            "confidence", "probabilities", "gradcam_overlay", "chat_history"]:
    # Initialize keys only if not already present in session_state
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize chat history with a greeting assistant message on first load
if st.session_state.chat_history is None:
    st.session_state.chat_history = [
        {"role": "assistant", "msg": "Hello! I'm here to support you. How are you feeling today?"}
    ]

# ==== Load Model ====
# (Owner: Md Kaif)
# Tech: PyTorch, Torchvision
# Explanation:
# - Decide device (GPU if available else CPU)
# - Create a ResNet-based model, modify final fully-connected layer for 5 classes
# - Load saved weights (path provided in repository)
# - Put model to eval mode for inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device
# NOTE: original code uses models.resnet50 with weights=None but filename says resnet101. Keeping same as your code.
model = models.resnet50(weights=None)  # base architecture (Kaif chose ResNet family for feature depth)
# Replace final FC with custom head for 5-class classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Linear(512, 5)
)

# Load trained weights from a .pth file. Map to device for compatibility.
# If file not found or shape mismatches, strict=False allows partial loading (useful during development).
model.load_state_dict(torch.load("resnet101_lung_model_320.pth", map_location=device), strict=False)
model.to(device)   # move model to selected device
model.eval()       # set model to evaluation mode (disables dropout, batchnorm tracking, etc.)

# Class names used to map predicted index to human-readable class
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# ==== Grad-CAM Setup ====
# (Owner: Md Kaif & Zuhaib for preprocessing of activation maps)
# Tech: PyTorch hooks, NumPy, OpenCV
# Explanation:
# - To compute Grad-CAM we need both forward activations and backward gradients of a target convolutional layer.
# - We register hooks on model.layer4[-1] (last block of layer4) to save activations and gradients.
# - Hooks run automatically when forward() and backward() are executed on the model.

gradients = None   # will hold gradients from backward hook
activations = None # will hold activations from forward hook

def save_gradient(module, grad_input, grad_output):
    """
    Backward hook: called during .backward() and receives gradient outputs.
    We store the gradient tensor globally for later Grad-CAM computation.
    (Owner: Kaif)
    """
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    """
    Forward hook: captures activations (feature maps) from the chosen convolutional block.
    (Owner: Kaif)
    """
    global activations
    activations = output

# Register hooks on the last block of layer4. These are attached once at app start.
# The hooks don't change model behavior; they just capture intermediate tensors.
model.layer4[-1].register_forward_hook(forward_hook)
# register_backward_hook is deprecated in newer torch; it still works in some versions.
model.layer4[-1].register_backward_hook(save_gradient)

# ==== Image Transform / Preprocessing Pipeline ====
# (Owner: Md Zuhaib Ali)
# Tech: Torchvision transforms, PIL
# Explanation:
# - Convert uploaded image to 3-channel grayscale (model expects 3 channels)
# - Resize to (224,224) compatible with ResNet input
# - Convert to tensor and normalize using ImageNet-like mean/std (keeps input scales consistent with pretrained features)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),          # convert single-channel X-ray to 3 channels
    transforms.Resize((224, 224)),                        # ResNet input size
    transforms.ToTensor(),                                # PIL -> Tensor (C,H,W), scales to [0,1]
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)   # normalization (approx ImageNet stats)
])

# ==== Layout: Columns (Chat | Main) ====
# (Owner: Shahil for layout + Kaif for integration)
# Tech: Streamlit
col_chat, col_main = st.columns([1, 7])  # narrow column for chatbot, wide column for main diagnostic UI

# ==== Chatbot UI (Left Column) ====
# (Owner: Mohammad Adil Siddiqui)
# Tech: DialoGPT (HuggingFace), PyTorch, Streamlit
with col_chat:
    st.markdown("### 💬 Talk to AI Assistant")  # small header for the chat widget

    # Initialize local chatbot instance (this wraps the DialoGPT/inference code).
    # LocalChatbot should provide an `ask()` method that takes user text and returns model reply.
    bot = LocalChatbot()

    # Display previous chat history stored in session_state
    for chat in st.session_state.chat_history:
        # st.chat_message used to present messages in chat-bubble style (assistant/user)
        with st.chat_message(chat["role"]):
            st.markdown(chat["msg"])

    # Capture user input from Streamlit's chat_input widget
    # This widget triggers a rerun when the user submits message
    if user_msg := st.chat_input("How are you feeling or confused about anything?"):
        # Append the user's message to session history for persistence
        st.session_state.chat_history.append({"role": "user", "msg": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Pass user message to chatbot and get reply (synchronous)
        # (Owner: Adil)
        reply = bot.ask(user_msg)   # LocalChatbot.ask should handle inference + text postprocessing
        st.session_state.chat_history.append({"role": "assistant", "msg": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# ==== Diagnostic UI (Right Column) ====
# (Owner: Kaif + Shahil)
with col_main:
    st.title("🫁 Lung Disease Detection using ResNet101")
    st.write("Upload chest X-ray and fill patient info to generate a report.")

    # Using a Streamlit form groups inputs and triggers single submission event.
    # (Owner: Shahil for fields design, Kaif for backend wiring)
    with st.form("input_form"):
        st.subheader("👤 Patient Details")
        c1, c2 = st.columns(2)   # two columns inside form for better layout

        # Patient detail inputs (persist via local variables; final values saved to session_state after prediction)
        name = c1.text_input("Name")                          # text input for patient name
        age = c2.number_input("Age", 1, 120)                  # numeric input with min/max
        gender = c1.selectbox("Gender", ["Male", "Female", "Other"])  # gender selection
        symptoms = c2.text_input("Symptoms (comma separated)")       # symptom notes

        # File uploader to accept chest X-ray images (jpg/jpeg/png)
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
        # Grad-CAM checkbox removed in this version for simplicity; we will always compute Grad-CAM when image processed.

        # Submit button triggers a rerun with `submitted=True`
        submitted = st.form_submit_button("🔎 Analyze")

    # ==== Prediction & Grad-CAM Execution Block ====
    # This block executes when: there's no prior prediction in session_state, the user submitted the form, and file uploaded.
    # (Owner: Kaif for model inference & Grad-CAM)
    if st.session_state.pred_class is None and submitted and uploaded_file:
        # Read uploaded file into PIL Image and convert to 'L' (grayscale) because X-rays are grayscale
        image = Image.open(uploaded_file).convert("L")  # PIL Image object

        # Apply preprocessing transform (Grayscale->3ch, resize, to tensor, normalize)
        input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim -> shape (1,3,224,224)

        # For Grad-CAM we need gradients wrt input activations; set requires_grad on input tensor
        input_tensor.requires_grad_()

        # Forward pass through the model (prediction)
        output = model(input_tensor)  # logits shape (1,5)
        # Convert logits -> softmax probabilities for human-readable confidences
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]  # numpy array shape (5,)
        # Predicted class index (argmax over logits)
        pred_index = torch.argmax(output, dim=1).item()
        pred_class = class_names[pred_index]  # map index -> class label
        confidence = probs[pred_index] * 100  # percent confidence

        # === Grad-CAM computation ===
        # 1) Zero gradients on model to clear old values
        model.zero_grad()

        # 2) Backpropagate the score of the predicted class to compute gradients wrt activations
        #    This triggers the registered backward hook to store gradients in `gradients`.
        output[0, pred_index].backward()

        # 3) Following Grad-CAM algorithm: compute pooled gradients across spatial dimensions for each channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # shape: [C]
        activation_maps = activations[0]  # activations from forward hook: shape [C, H, W]

        # 4) Weight the activation maps by the pooled gradients
        for i in range(activation_maps.shape[0]):
            activation_maps[i] *= pooled_gradients[i]

        # 5) Create raw heatmap by averaging the weighted activations across channels
        heatmap = torch.mean(activation_maps, dim=0).cpu().detach().numpy()
        # Keep only positive values (ReLU)
        heatmap = np.maximum(heatmap, 0)
        # Normalize to [0,1]
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        # 6) Resize heatmap to image size and convert to color map for overlay
        heatmap = cv2.resize(heatmap, (224, 224))  # resize to same size as display image
        heatmap = np.uint8(255 * heatmap)          # convert to uint8 for color mapping

        # Convert grayscale PIL image to BGR for OpenCV overlay (resize first to 224x224)
        orig_image_resized = np.array(image.resize((224, 224)))
        # If grayscale image, convert to BGR (3 channels) so addWeighted works
        orig_bgr = cv2.cvtColor(orig_image_resized, cv2.COLOR_GRAY2BGR)

        # Apply colormap (JET) to heatmap and overlay with original image to highlight areas
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_bgr, 0.5, colored_heatmap, 0.5, 0)

        # ==== Save results into session_state so the UI can use them across reruns ====
        st.session_state.update({
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "image": image,
            "pred_class": pred_class,
            "confidence": confidence,
            "probabilities": probs,
            "gradcam_overlay": overlay
        })

    # ==== Display Results if Available ====
    if st.session_state.pred_class is not None:
        # Show the original uploaded image (PIL) in the UI
        st.image(st.session_state.image, caption="📷 Uploaded Image", use_container_width=True)
        # Show prediction as a success message
        st.success(f"Prediction: {st.session_state.pred_class}")
        # Show confidence percentage
        st.write(f"Confidence: {st.session_state.confidence:.2f}%")
        # Display the Grad-CAM overlay (converted numpy BGR image shown with Streamlit)
        st.image(st.session_state.gradcam_overlay, caption="📍 Infected Area Highlight", use_container_width=True)

        # Show class-wise probabilities in a readable list
        st.subheader("📊 Class Probabilities:")
        for i, prob in enumerate(st.session_state.probabilities):
            st.write(f"- {class_names[i]}: {prob * 100:.2f}%")

        # ==== Download PDF (Report) Generation ====
        # (Owner: Mohammad Shahil)
        # Tech: FPDF
        # Explanation:
        # - Create structured PDF with patient info, diagnosis, confidence and class probabilities
        # - Return binary content for Streamlit download_button
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()

            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "LUNG DISEASE REPORT", ln=True, align='C')
            pdf.line(10, 20, 200, 20)  # horizontal line for separation
            pdf.ln(10)

            # Patient Info Section
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, "PATIENT INFO", ln=True)
            pdf.set_font("Arial", "", 12)
            # Insert patient details from session_state
            pdf.cell(0, 8, f"Name: {st.session_state.name}", ln=True)
            pdf.cell(0, 8, f"Age: {st.session_state.age}", ln=True)
            pdf.cell(0, 8, f"Gender: {st.session_state.gender}", ln=True)
            pdf.multi_cell(0, 8, f"Symptoms: {st.session_state.symptoms}")
            pdf.ln(4)

            # Diagnosis section
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, f"DIAGNOSIS: {st.session_state.pred_class}", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Confidence: {st.session_state.confidence:.2f}%", ln=True)
            pdf.ln(4)

            # Class probabilities
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "CLASS PROBABILITIES:", ln=True)
            for i, prob in enumerate(st.session_state.probabilities):
                pdf.cell(0, 8, f"{class_names[i]}: {prob * 100:.2f}%", ln=True)

            pdf.ln(5)
            # Footer with generation date
            pdf.set_font("Arial", "I", 9)
            pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%d %B %Y')}", ln=True)

            # Return PDF binary data; Streamlit download_button expects bytes-like object
            return pdf.output(dest='S').encode('latin-1', errors='replace')

        # Generate PDF and present download button (Shahil)
        pdf_data = generate_pdf()
        st.download_button("📥 Download PDF Report", data=pdf_data, file_name="Lung_Report.pdf", mime="application/pdf")

    # ==== Reset Button ====
    # (Owner: Shahil + Kaif)
    # Clears session and reruns app to initial state
    if st.button("🔄 Reset"):
        st.session_state.clear()
        # Force a rerun to reflect cleared state immediately
        st.experimental_rerun()

# ============================================
# END of Script
# ============================================

# ========== Additional Notes for Viva (include when explaining code) ==========
# - Important Streams:
#   * UI: Streamlit (owner: Shahil + Kaif)
#   * Model + Grad-CAM: PyTorch (owner: Kaif)
#   * Preprocessing: PIL/OpenCV/Torchvision (owner: Zuhaib)
#   * Chatbot: Local DialoGPT wrapper (owner: Adil)
#   * PDF/report: FPDF (owner: Shahil)
#
# - Execution Flow Summary (one-liner for viva):
#   "User uploads X-ray -> preprocessing -> model forward -> prediction -> backward pass for Grad-CAM -> heatmap overlay -> show result & generate PDF. Chatbot runs separately for user support."
#
# - Multiple internal calls to note:
#   * model(input_tensor) -> forward pass (activations saved by forward hook)
#   * output[0, pred_index].backward() -> backward pass (gradients saved by backward hook)
#   * Hooks fire automatically during forward/backward; we only call forward/backward explicitly.
#
# - Troubleshooting tips:
#   * If Grad-CAM heatmap is blank, ensure `activations` and `gradients` are populated (check hooks).
#   * If model weight loading fails, check path and `strict` flag; strict=False helps during development.
#   * For speed, use CUDA (if available) and cache model load (in production consider st.cache_resource).
#
# ==================================================================================
