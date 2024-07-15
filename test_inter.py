import cv2
import numpy as np
import logging 
from skimage import img_as_ubyte
# Define the available classes and corresponding colors for visualization
classes = {
    1: (255, 0, 0),  # Blue for class 1 
    2: (0, 255, 0),  # Green for class 2 
    3: (0, 0, 255)   # Red for class 3 
}
current_class = 1  # Default class to start with

def load_image(image_path):
    """Load an image from the specified file path with error handling."""
    # First, check if the provided image path is empty
    if image_path == "":
        raise ValueError("Empty string provided as input. Please enter a valid path.")

    try:
        # Try to read the image from the given path
        image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            raise FileNotFoundError(f"Error: Could not load image from {image_path}. Please check the path and try again.")
        
        # Return the loaded image
        return image
    
    # Handle specific exceptions
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")
        raise

def grabcut_segmentation(image, rect):
    """Perform grabCut segmentation on the image.     
    This function uses the grabCut algorithm to segment the foreground of an image from its background.
    The algorithm takes an initial rectangular region  and iteratively refines the segmentation."""
    
    #Initialize the mask with zeros
    mask = np.zeros(image.shape[:2], np.uint8)
    # Create models for the background and foreground.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply the grabCut algorithm.
    # This modifies the mask, bgd_model, and fgd_model in place.
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask where foreground pixels are marked with 1 and background pixels are marked with 0.
    # The grabCut algorithm labels pixels with 0 (definitely background), 1 (definitely foreground), 
    #The segmented image is created by multiplying the original image with the binary mask.
    #This operation blackens out the background pixels and retains the foreground pixels
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image * mask2[:, :, np.newaxis]
    
    return segmented, mask2

def label_image(image_path, output_path):
    """
    Label regions in an image using grabCut segmentation and user input.

    """
    global current_class
    # Load the image
    image = load_image(image_path)
    labeled_image = np.zeros(image.shape[:2], dtype=np.uint8)
    #print("Press '1', '2', or '3' to select class. Press 'ESC' to save and exit.")
    label = int(input("Please enter 1, 2, or 3 to label the region (1:Blue, 2:Green, 3:Red): "))
    if label not in classes:
        print("Invalid label. Please enter 1, 2, or 3.")
        return
    current_class = label
        
    def draw_rectangle(event, x, y, flags, param):
        """Mouse callback function to draw rectangles and label the image regions."""
        global drawing, ix, iy, current_class
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            print(f"Started drawing at: ({ix}, {iy})")
        # elif event == cv2.EVENT_MOUSEMOVE:
        #     if drawing:
        #         img_copy = image.copy()
        #         cv2.rectangle(img_copy, (ix, iy), (x, y), classes[current_class], 1, lineType=4)
        #         cv2.imshow('image', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), classes[current_class], 1)
            x_min, x_max = min(ix, x), max(ix, x)
            y_min, y_max = min(iy, y), max(iy, y)
            print(f"Finished drawing from: ({ix}, {iy}) to ({x}, {y})")
            print(f"Labeling region: x({x_min}, {x_max}), y({y_min}, {y_max}) with class {current_class}")
            
            # Perform GrabCut segmentation for the drawn rectangle
            rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            segmented, mask2 = grabcut_segmentation(image, rect)
            
            # Assign the current class to the segmented region
            labeled_image[y_min:y_max, x_min:x_max] = np.where(mask2[y_min:y_max, x_min:x_max] == 1, current_class, labeled_image[y_min:y_max, x_min:x_max])

    drawing = False
    ix, iy = -1, -1

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)


    
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'ESC' to exit the 
            break
        # elif k == ord('1'):
        #     current_class = 1
        #     print("Current class: 1 (Blue)")
        # elif k == ord('2'):
        #     current_class = 2
        #     print("Current class: 2 (Green)")
        # elif k == ord('3'):
        #     current_class = 3
        #     print("Current class: 3 (Red)")

    cv2.destroyAllWindows()

    # Visualize the labeled image for verification
    labeled_image_color = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
    for cls, color in classes.items():
        labeled_image_color[labeled_image == cls] = color
    cv2.imshow('Labeled Image', labeled_image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the labeled image
    print(f"Saving labeled image to {output_path}")
    #labeled_image = labeled_image.todtype()
   # print(labeled_image)
    #labeled_image = labeled_image*255
   # labeled_image = img_as_ubyte(labeled_image)
    #print("----------")
    #print(labeled_image)
    #labeled_image_color = Image.fromarray(labeled_image_color)
    #result = cv2.normalize(labeled_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_path, labeled_image_color)

if __name__ == "__main__":
    image_source_path = input("Enter the absolute path of source image file:")
    image_destination_path = input("Enter the absolute path of target image file:")
    #label_image('C:/Users/Gautham/Datasets/simpson.jpg', 'C:/Users/Gautham/Datasets/labeled_image.png') 
    label_image(image_source_path,image_destination_path)