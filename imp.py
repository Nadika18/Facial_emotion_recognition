import cv2
import torch
from torchvision import transforms
from model3 import EmotionCNN, emotion_categories

# Load the trained model
model = EmotionCNN(num_classes=8)
model.load_state_dict(torch.load("model4.h5"))
model.eval()

# Define a transform for preprocessing webcam frames
preprocess = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the indices of negative emotions
negative_emotion_indices = [2, 4, 6]  # Sadness, Afraid, Angry

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

# Main loop for emotion detection
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break
    
    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension
    
    # Get emotion probabilities from the model
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    # Calculate the combined probability of negative emotions
    negative_emotion_prob = sum(probabilities[i] for i in negative_emotion_indices)
    if negative_emotion_prob > 0.5:
        # Sound an alarm (you can replace this with your preferred alarm mechanism)
        print("Negative emotions detected! Sounding an alarm...")
        # Implement your alarm mechanism here
        
    # Display emotion probabilities on the frame
    for i, (emotion, prob) in enumerate(zip(emotion_categories.values(), probabilities)):
        cv2.putText(frame, f"{emotion}: {prob:.2f}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Emotion Detection", frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
