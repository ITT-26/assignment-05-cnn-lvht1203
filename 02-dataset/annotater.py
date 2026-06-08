import cv2
import json
import os

IMAGE_DIR = "02-dataset/Dataset"
OUTPUT_JSON = "annot-thu.json"

VALID_LABELS = ["like", "dislike", "stop", "rock", "peace"]

annotations = {}
current_points = []
current_image = None
display_image = None
current_filename = None


def get_label_from_filename(filename):
    name = filename.lower()
    for label in VALID_LABELS:
        if name.startswith(label):
            return label
    return input(f"Label for {filename}: ").strip()


def mouse_callback(event, x, y, flags, param):
    global current_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(display_image, (x, y), 6, (0, 0, 255), -1)

        if len(current_points) == 2:
            x1, y1 = current_points[0]
            x2, y2 = current_points[1]

            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Annotater", display_image)


def annotate_image(filepath):
    global current_points, current_image, display_image, current_filename

    current_filename = os.path.basename(filepath)
    current_points = []

    current_image = cv2.imread(filepath)
    if current_image is None:
        print(f"Could not read image: {filepath}")
        return None

    display_image = current_image.copy()
    h, w = current_image.shape[:2]

    cv2.imshow("Annotater", display_image)
    cv2.setMouseCallback("Annotater", mouse_callback)

    print(f"\nAnnotating: {current_filename}")
    print("Click TOP-LEFT and BOTTOM-RIGHT of the hand bbox.")
    print("Press 's' to save this annotation.")
    print("Press 'r' to reset points.")
    print("Press 'q' to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):
            current_points = []
            display_image = current_image.copy()
            cv2.imshow("Annotater", display_image)
            print("Reset points.")

        elif key == ord("s"):
            if len(current_points) != 2:
                print("You need exactly 2 points before saving.")
                continue

            x1, y1 = current_points[0]
            x2, y2 = current_points[1]

            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            bbox = [
                x_min / w,
                y_min / h,
                (x_max - x_min) / w,
                (y_max - y_min) / h
            ]

            label = get_label_from_filename(current_filename)

            image_id = os.path.splitext(current_filename)[0]

            return image_id, {
                "bboxes": [bbox],
                "labels": [label]
            }

        elif key == ord("q"):
            return "QUIT", None


def main():
    global annotations

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = {}

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    for filename in image_files:
        image_id = os.path.splitext(filename)[0]

        if image_id in annotations:
            print(f"Skipping already annotated image: {filename}")
            continue

        result_id, result_data = annotate_image(os.path.join(IMAGE_DIR, filename))

        if result_id == "QUIT":
            break

        annotations[result_id] = result_data

        with open(OUTPUT_JSON, "w") as f:
            json.dump(annotations, f, indent=4)

        print(f"Saved annotation for {filename}")

    cv2.destroyAllWindows()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"\nDone. Annotation file saved as {OUTPUT_JSON}")


if __name__ == "__main__":
    main()