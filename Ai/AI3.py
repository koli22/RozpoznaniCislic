import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (280, 280)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DRAWING_AREA = pygame.Rect(40, 40, 200, 200)

# Load the saved model
model = load_model('digit_recognition_model.h5')
print("Model loaded from digit_recognition_model.h5")

# Function to preprocess the drawn image
def preprocess_image(surface):
    # Convert Pygame surface to PIL Image
    img = pygame.surfarray.array3d(surface)
    img = Image.fromarray(img)

    # Convert to grayscale and resize
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((25, 25))  # Resize to match model's expected sizing

    # Convert back to numpy array
    img_array = np.array(img)
    img_array = img_array.reshape(-1, 25, 25, 1)/255.0  # Reshape for model input
    #img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array

# Set up the screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Digit Recognition")

# Font for displaying predicted number
font = pygame.font.Font(None, 36)

# Variables
drawing = False
last_pos = None
predicted_number = None
drawing_img = None

screen.fill(WHITE)
drawing_area = pygame.Surface((200, 200))
drawing_area.fill(WHITE)
pygame.draw.rect(drawing_area, BLACK, drawing_area.get_rect(), 1)
screen.blit(drawing_area, (40, 40))


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if DRAWING_AREA.collidepoint(event.pos):
                drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mouse_x, mouse_y = event.pos
                if DRAWING_AREA.collidepoint(event.pos) and last_pos is not None:
                    pygame.draw.line(screen, BLACK, last_pos, (mouse_x, mouse_y), 25)
                    last_pos = (mouse_x, mouse_y)
                    
    # Clear the screen when 'c' is pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_c]:
        screen.fill(WHITE)
        pygame.draw.rect(drawing_area, BLACK, drawing_area.get_rect(), 1)
        predicted_number = None

    # Get the drawing from the screen, preprocess it, and predict
    if keys[pygame.K_p] and not drawing:
        drawing_area = pygame.Surface((200, 200))
        drawing_area.fill(WHITE)
        pygame.draw.rect(drawing_area, BLACK, drawing_area.get_rect(), 1)
        screen.blit(drawing_area, (40, 40))
        pygame.display.update()

        # Preprocess the drawn image
        drawing_img = preprocess_image(drawing_area)

        # Display the processed image on the screen
        # Predict using the loaded model
        prediction = model.predict(drawing_img)
        predicted_number = np.argmax(prediction)

    # Display the predicted number
    if predicted_number is not None:
        text = font.render(f"Predicted Number: {predicted_number}", True, BLACK)
        screen.blit(text, (20, 10))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

