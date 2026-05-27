import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from keras.datasets import mnist

# ============================================
# NEURAL NETWORK CLASS
# ============================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        exp_z = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.A2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.A2
    
    def predict_class(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)


# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    nn = NeuralNetwork(784, 128, 10)
    data = np.load('mnist_weights.npz')
    nn.W1 = data['W1']
    nn.b1 = data['b1']
    nn.W2 = data['W2']
    nn.b2 = data['b2']
    return nn


# ============================================
# ENHANCED PREPROCESSING WITH MULTIPLE ATTEMPTS
# ============================================
def preprocess_image_multiple(image):
    """
    Try multiple preprocessing variations and pick the best
    """
    variations = []
    
    # Variation 1: Standard preprocessing
    img_flat1, img_28_1 = standard_preprocess(image)
    variations.append(('Standard', img_flat1, img_28_1))
    
    # Variation 2: Different threshold (lighter)
    img_flat2, img_28_2 = threshold_preprocess(image, percentile=50)
    variations.append(('Light threshold', img_flat2, img_28_2))
    
    # Variation 3: Different threshold (darker)
    img_flat3, img_28_3 = threshold_preprocess(image, percentile=70)
    variations.append(('Dark threshold', img_flat3, img_28_3))
    
    # Variation 4: No blur
    img_flat4, img_28_4 = no_blur_preprocess(image)
    variations.append(('Sharp', img_flat4, img_28_4))
    
    # Variation 5: Thinner lines (morphological erosion)
    img_flat5, img_28_5 = thin_lines_preprocess(image)
    variations.append(('Thin lines', img_flat5, img_28_5))
    
    return variations


def standard_preprocess(image):
    """Standard preprocessing"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((280, 280), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    threshold = np.percentile(img_array, 60)
    img_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
    
    # Crop to digit
    coords = np.argwhere(img_array == 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        img_array = img_array[y_min:y_max+1, x_min:x_max+1]
    
    # Make square
    size = max(img_array.shape)
    squared = np.ones((size, size)) * 255
    y_offset = (size - img_array.shape[0]) // 2
    x_offset = (size - img_array.shape[1]) // 2
    squared[y_offset:y_offset+img_array.shape[0], x_offset:x_offset+img_array.shape[1]] = img_array
    
    # Resize to 28x28
    img_pil = Image.fromarray(squared.astype(np.uint8))
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    # Blur
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_array = np.array(img_pil)
    
    # Normalize and invert
    img_array = img_array / 255.0
    img_array = 1 - img_array
    
    img_flat = img_array.reshape(1, 784)
    return img_flat, img_array


def threshold_preprocess(image, percentile=60):
    """Same as standard but with adjustable threshold"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((280, 280), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    threshold = np.percentile(img_array, percentile)
    img_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
    
    coords = np.argwhere(img_array == 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        img_array = img_array[y_min:y_max+1, x_min:x_max+1]
    
    size = max(img_array.shape)
    squared = np.ones((size, size)) * 255
    y_offset = (size - img_array.shape[0]) // 2
    x_offset = (size - img_array.shape[1]) // 2
    squared[y_offset:y_offset+img_array.shape[0], x_offset:x_offset+img_array.shape[1]] = img_array
    
    img_pil = Image.fromarray(squared.astype(np.uint8))
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_array = np.array(img_pil)
    
    img_array = img_array / 255.0
    img_array = 1 - img_array
    
    img_flat = img_array.reshape(1, 784)
    return img_flat, img_array


def no_blur_preprocess(image):
    """No blur, keep sharp edges"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((280, 280), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    threshold = np.percentile(img_array, 60)
    img_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
    
    coords = np.argwhere(img_array == 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        img_array = img_array[y_min:y_max+1, x_min:x_max+1]
    
    size = max(img_array.shape)
    squared = np.ones((size, size)) * 255
    y_offset = (size - img_array.shape[0]) // 2
    x_offset = (size - img_array.shape[1]) // 2
    squared[y_offset:y_offset+img_array.shape[0], x_offset:x_offset+img_array.shape[1]] = img_array
    
    img_pil = Image.fromarray(squared.astype(np.uint8))
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    # NO BLUR here
    img_array = img_array / 255.0
    img_array = 1 - img_array
    
    img_flat = img_array.reshape(1, 784)
    return img_flat, img_array


def thin_lines_preprocess(image):
    """Make lines thinner to better match MNIST"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((280, 280), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    threshold = np.percentile(img_array, 55)
    img_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
    
    # Simple thinning: erode
    from scipy.ndimage import binary_erosion
    from scipy.ndimage import generate_binary_structure
    
    structure = generate_binary_structure(2, 1)
    binary = img_array == 0
    eroded = binary_erosion(binary, structure=structure, iterations=1)
    img_array = np.where(eroded, 0, 255).astype(np.uint8)
    
    coords = np.argwhere(img_array == 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        img_array = img_array[y_min:y_max+1, x_min:x_max+1]
    
    size = max(img_array.shape)
    squared = np.ones((size, size)) * 255
    y_offset = (size - img_array.shape[0]) // 2
    x_offset = (size - img_array.shape[1]) // 2
    squared[y_offset:y_offset+img_array.shape[0], x_offset:x_offset+img_array.shape[1]] = img_array
    
    img_pil = Image.fromarray(squared.astype(np.uint8))
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.3))
    img_array = np.array(img_pil)
    
    img_array = img_array / 255.0
    img_array = 1 - img_array
    
    img_flat = img_array.reshape(1, 784)
    return img_flat, img_array


# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Upload a photo of a handwritten digit (0-9)")

# Load model
try:
    nn = load_model()
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", width=250)
    
    with st.spinner("Processing..."):
        variations = preprocess_image_multiple(image)
    
    st.markdown("### 🔄 Trying Multiple Processing Methods")
    st.markdown("The model tries 5 different ways to process your image and picks the best result:")
    
    results = []
    
    for name, img_flat, img_28 in variations:
        pred = nn.predict_class(img_flat)
        probs = nn.forward(img_flat)
        confidence = probs[0][pred[0]]
        results.append((name, pred[0], confidence, img_28, probs[0]))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Display all results
    cols = st.columns(len(variations))
    for idx, (name, pred, conf, img_28, _) in enumerate(results):
        with cols[idx]:
            st.image(img_28, caption=f"{name}\nPredicted: {pred}\n{conf:.1%}", width=100)
    
    # Best result
    best_name, best_pred, best_conf, best_img, best_probs = results[0]
    
    st.markdown("---")
    st.markdown("## 🎯 Best Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Digit", f"**{best_pred}**")
    with col2:
        st.metric("Confidence", f"{best_conf:.2%}")
    with col3:
        if best_conf > 0.8:
            st.success("✅ High confidence")
        elif best_conf > 0.6:
            st.warning("⚠️ Medium confidence")
        else:
            st.error("❌ Low confidence - try a clearer image")
    
    # Show confidence for all digits
    st.markdown("### Confidence for each digit")
    prob_cols = st.columns(10)
    for i in range(10):
        with prob_cols[i]:
            st.metric(str(i), f"{best_probs[i]:.1%}")
    
    if best_pred != 5:
        st.info("💡 **Tip**: If you wrote a 5 but got a different prediction, try:\n- Writing larger and clearer\n- Making sure the top loop of 5 is open, not closed\n- Ensuring good lighting and contrast")

# st.sidebar.markdown("## Tips for 5 vs 2")
# st.sidebar.markdown("""
# The model sometimes confuses 5 and 2 because:
# - A 5 with a closed loop looks like a 2
# - A 2 with a sharp top looks like a 5

# **To write a better 5:**
# - Keep the top loop OPEN
# - Make the horizontal line clear
# - Write larger
# - Use high contrast (black ink, white paper)
# """)