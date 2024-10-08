import cv2
import numpy as np

# Parameters
car_width = 2.0  # in meters
car_length = 4.0  # in meters
distance_between_cars = 0.5  # in meters
aisle_width = 3.0  # in meters

# Scale factor to convert meters to pixels (adjust based on your image)
scale = 50  # Example scale: 1 meter = 50 pixels

# Convert dimensions to pixels
car_width_px = int(car_width * scale)
car_length_px = int(car_length * scale)
distance_between_cars_px = int(distance_between_cars * scale)
aisle_width_px = int(aisle_width * scale)

# Load the binary mask image of the parking lot
image = cv2.imread('parking_lot.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or unable to load")

# Detect the contour of the parking lot
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("No contours found in the image")
parking_lot_contour = contours[0]

# Create a blank canvas to draw the grid
canvas = np.zeros_like(image)

# Iterate over the parking lot area and place car grids
x, y, w, h = cv2.boundingRect(parking_lot_contour)

for row in range(y, y + h, car_length_px + distance_between_cars_px + aisle_width_px):
    for col in range(x, x + w, car_width_px + distance_between_cars_px):
        # Create a rectangle representing the car space
        car_space = np.array([
            [col, row],
            [col + car_width_px, row],
            [col + car_width_px, row + car_length_px],
            [col, row + car_length_px]
        ], dtype=np.int32)
        
        # Check if the car space is within the parking lot contour
        if cv2.pointPolygonTest(parking_lot_contour, (col, row), False) >= 0:
            # Draw the car space on the canvas
            cv2.polylines(canvas, [car_space], isClosed=True, color=255, thickness=2)

# Show the resulting grid
cv2.imshow('Parking Grid', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('parking_grid.png', canvas)
