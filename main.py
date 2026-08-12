import os
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

MODEL_ID = "ppes-kaxsi/8"
VIOLATION_CLASSES = {"no_helmet", "no_vest", "no_mask", "no_goggles", "no_shoes", "no_glove"}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite("frame.jpg", frame)
    result = CLIENT.infer("frame.jpg", model_id=MODEL_ID)

    for pred in result.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        label = pred["class"]
        conf = pred["confidence"]

        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        is_violation = label in VIOLATION_CLASSES
        color = (0, 0, 255) if is_violation else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if is_violation:
            print(f"[ALERT] {label} ({conf:.2f})")

    cv2.imshow("PPE Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()