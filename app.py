import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image
import io

# Define the model class (same as in your notebook)
class PlantDiseaseModel(nn.Module):
    def __init__(self, classes=3):  # 3 classes: Healthy, Powdery, Rust
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet34(pretrained=True)
        for parameter in self.model.parameters():
            parameter.require_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=classes),
        )

    def forward(self, image):
        output = self.model(image)
        return output

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        # Try loading as a regular PyTorch model first
        model = PlantDiseaseModel(classes=3)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except:
        try:
            # If that fails, try loading as TorchScript
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

def preprocess_image(image, image_shape=(128, 128)):
    """Preprocess the uploaded image for inference"""
    # Convert PIL image to numpy array
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    
    # Apply transformations
    transform = A.Compose([
        A.Resize(height=image_shape[0], width=image_shape[1]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_array)
    image_tensor = transformed["image"].float().unsqueeze(0)
    
    return image_tensor

def predict_image(model, image_tensor, class_names):
    """Make prediction on the preprocessed image"""
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        confidence_score = torch.max(probabilities).item()
        
    predicted_class = class_names[predicted_class_index]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(class_names):
        class_probabilities[class_name] = probabilities[0][i].item()
    
    return predicted_class, confidence_score, class_probabilities

def main():
    st.set_page_config(
        page_title="Plant Disease Classifier",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("🌿 Plant Disease Classification App")
    st.markdown("Upload an image of a plant leaf to detect if it's healthy or diseased!")
    
    # Model file path (fixed in project directory)
    model_path = "plant_disease_model.pt"
    
    # Sidebar for app configuration
    st.sidebar.header("Configuration")
    
    # Class names (based on your dataset)
    class_names = ["Healthy", "Powdery", "Rust"]
    
    # Model status in sidebar
    if st.sidebar.button("🔄 Refresh Model"):
        st.cache_resource.clear()
    
    st.sidebar.markdown(f"**Model:** `{model_path}`")
    
    # Image shape configuration
    img_height = st.sidebar.slider("Image Height", 64, 512, 128)
    img_width = st.sidebar.slider("Image Width", 64, 512, 128)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.write(f"**Image size:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")
    
    with col2:
        st.header("Prediction Results")
        
        if uploaded_file is not None:
            # Add predict button
            predict_button = st.button("🔮 Predict Disease", type="primary", use_container_width=True)
            
            # Initialize session state for storing results
            if 'prediction_made' not in st.session_state:
                st.session_state.prediction_made = False
                st.session_state.prediction_results = None
            
            if predict_button:
                try:
                    # Reset prediction state
                    st.session_state.prediction_made = False
                    st.session_state.prediction_results = None
                    
                    # Load model from project directory
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                    
                    if model is not None:
                        st.sidebar.success("✅ Model loaded successfully!")
                        
                        # Preprocess image
                        with st.spinner("Processing image..."):
                            image_tensor = preprocess_image(image, (img_height, img_width))
                        
                        # Make prediction
                        with st.spinner("Making prediction..."):
                            predicted_class, confidence, class_probs = predict_image(
                                model, image_tensor, class_names
                            )
                        
                        # Store results in session state
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'class_probs': class_probs
                        }
                        st.session_state.prediction_made = True
                        
                    else:
                        st.sidebar.error("❌ Model loading failed!")
                        st.error(f"❌ Could not load model from `{model_path}`. Please ensure the model file exists in the project directory.")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {str(e)}")
                    st.error("Please check your model file and try again.")
            
            # Display results if prediction has been made
            if st.session_state.prediction_made and st.session_state.prediction_results:
                results = st.session_state.prediction_results
                predicted_class = results['predicted_class']
                confidence = results['confidence']
                class_probs = results['class_probs']
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Main prediction with larger text
                st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
                st.markdown(f"### 📊 Confidence: **{confidence:.2%}**")
                
                # Progress bar for overall confidence
                st.progress(confidence)
                
                # Detailed probabilities
                st.markdown("### 📈 Class Probabilities:")
                for class_name, prob in class_probs.items():
                    col_class, col_prob, col_bar = st.columns([1, 1, 2])
                    with col_class:
                        # Add emoji based on class
                        emoji = "✅" if class_name == "Healthy" else "🦠" if class_name == "Powdery" else "🔴"
                        st.write(f"{emoji} {class_name}")
                    with col_prob:
                        st.write(f"{prob:.2%}")
                    with col_bar:
                        st.progress(prob)
                
                # Health status interpretation
                st.markdown("### 🔍 Interpretation:")
                if predicted_class == "Healthy":
                    st.success("🌱 The plant appears to be healthy! No signs of disease detected.")
                elif predicted_class == "Powdery":
                    st.warning("⚠️ Powdery mildew detected! This fungal disease appears as white powdery spots on leaves.")
                elif predicted_class == "Rust":
                    st.error("🚨 Rust disease detected! This appears as rust-colored or brown spores on plant surfaces.")
                
                # Confidence interpretation
                st.markdown("### 🎯 Confidence Level:")
                if confidence > 0.9:
                    st.success("🎯 **Very High Confidence** - The model is very sure about this prediction!")
                elif confidence > 0.7:
                    st.info("👍 **High Confidence** - The model is confident about this prediction!")
                elif confidence > 0.5:
                    st.warning("🤔 **Moderate Confidence** - Consider getting a second opinion or using a clearer image.")
                else:
                    st.error("⚠️ **Low Confidence** - The image might be unclear or the condition might be ambiguous. Try uploading a different image.")
                
                # Add feedback section
                st.markdown("---")
                st.markdown("### 💬 Was this prediction correct?")
                feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                
                with feedback_col1:
                    if st.button("✅ Correct", key="correct_feedback"):
                        st.success("Thank you for the feedback! 🎉")
                        st.balloons()
                
                with feedback_col2:
                    if st.button("❌ Incorrect", key="incorrect_feedback"):
                        st.error("Thank you for the feedback. This helps improve the model! 📝")
                
                with feedback_col3:
                    if st.button("🤷 Not Sure", key="unsure_feedback"):
                        st.info("Thank you for the feedback. Consider consulting an expert! 👨‍🔬")
                
            elif uploaded_file is not None and not st.session_state.prediction_made:
                st.info("👆 Click the **Predict Disease** button above to analyze the uploaded image!")
        
        else:
            st.info("👈 Please upload an image first to get started!")
    
    # Information section
    with st.expander("ℹ️ About Plant Diseases"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🦠 Powdery Mildew:**
            - Fungal disease affecting many plants
            - Appears as white powdery spots on leaves
            - Thrives in high humidity and moderate temperatures
            - Can significantly reduce crop yields
            """)
        
        with col2:
            st.markdown("""
            **🔴 Rust Disease:**
            - Caused by pathogenic fungi (Pucciniales order)
            - Appears as rust-colored or brown spores
            - Highly specialized plant pathogens
            - Can cause stunted growth and chlorosis
            """)
    
    # Usage instructions
    with st.expander("📝 How to Use"):
        st.markdown("""
        **📝 How to Use:**
        1. **Model Setup**: Ensure `plant_disease_model.pt` is in your project directory
        2. **Upload Image**: Choose a clear image of a plant leaf (PNG, JPG, or JPEG)
        3. **Wait for Processing**: The app will automatically load the model and make predictions
        4. **View Results**: Check the predicted class, confidence score, and detailed probabilities
        5. **Interpret Results**: Use the interpretation guide to understand the plant's health status
        
        **Tips for better results:**
        - Use clear, well-lit images
        - Focus on the leaf surface
        - Avoid blurry or low-resolution images
        - Ensure the leaf fills most of the frame
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Plant Disease Classification App | Built with Streamlit 🚀"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()