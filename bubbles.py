import cv2
import numpy as np
import os


def preprocess_bubble_sheet(image_path, output_dir="output_bubbles"):
    """
    Preprocess the bubble sheet image by performing the following steps:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Detect edges using Canny
    4. Find the largest contour (assuming it's the bubble sheet)
    5. Apply perspective transform if needed
    6. Save all intermediate images to the output directory

    Args:
    - image_path (str): Path to the input image
    - output_dir (str): Directory where processed images will be saved

    Returns:
    - warped (numpy array): The perspective-corrected image
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image. Check the file path.")
        return None

    cv2.imwrite(f"{output_dir}/01_original.png", image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/02_gray.png", gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/03_blurred.png", blurred)

    # Apply edge detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(f"{output_dir}/04_edges.png", edges)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found!")
        return None

    # Draw all contours
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/05_all_contours.png", contour_img)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour
    largest_contour_img = image.copy()
    cv2.drawContours(largest_contour_img, [largest_contour], -1, (0, 0, 255), 3)
    cv2.imwrite(f"{output_dir}/06_largest_contour.png", largest_contour_img)

    # Apply perspective transform
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Get width and height of the detected box
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Define destination points
    dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")

    # Get transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    cv2.imwrite(f"{output_dir}/07_warped.png", warped)

    print(f" Preprocessing completed! Images saved in {output_dir}")
    return warped


def detect_marked_bubbles(output_dir="output_bubbles"):
    """
    Loads the saved warped image, detects marked bubbles, and colors them green.

    Args:
    - output_dir (str): Directory where the warped image is saved.

    Returns:
    - result (numpy array): Image with detected bubbles colored in green.
    """

    # List files in the output directory to debug
    available_files = os.listdir(output_dir)
    print(f" Available files in {output_dir}: {available_files}")

    # Correctly set the path to the warped image
    warped_path = os.path.join(output_dir, "07_warped.png")  

    if not os.path.exists(warped_path):
        raise FileNotFoundError(f" Warped image not found in {output_dir}! Check the file names above.")

    # Load the warped image
    warped_image = cv2.imread(warped_path)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect marked bubbles
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the warped image to draw on it
    result = warped_image.copy()

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter small contours (to remove noise)
        if w > 15 and h > 15:
            # Calculate the fill ratio
            total_area = w * h
            filled_area = cv2.contourArea(contour)
            fill_ratio = filled_area / total_area

            # If the filled area is significant, it's a marked bubble
            if fill_ratio > 0.3:
                cv2.drawContours(result, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # Fill with green

    # Save the final result
    marked_path = os.path.join(output_dir, "marked_bubbles.png")
    cv2.imwrite(marked_path, result)

    print(f" Marked bubbles detected! Image saved at {marked_path}")
    return result

# Example usage
image_path = './answer_sheets/bubbles.png'
#preprocess_bubble_sheet(image_path)
detect_marked_bubbles()



