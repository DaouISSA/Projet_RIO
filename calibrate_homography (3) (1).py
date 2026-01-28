import cv2, json, numpy as np

CAM_INDEX = 0
OUT_W, OUT_H = 900, 600  # taille de la vue du dessus

pts = []

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))

cap = cv2.VideoCapture(CAM_INDEX)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Can't read camera frame")

clone = frame.copy()
cv2.namedWindow("calib")
cv2.setMouseCallback("calib", on_mouse)

while True:
    img = clone.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, str(i+1), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(img, "Click 4 corners (TL, TR, BR, BL). Press S to save.", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("calib", img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('r'):
        pts = []
    if k == ord('s') and len(pts) == 4:
        src = np.array(pts, dtype=np.float32)
        dst = np.array([[0,0],[OUT_W,0],[OUT_W,OUT_H],[0,OUT_H]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        data = {"src_pts": pts, "out_w": OUT_W, "out_h": OUT_H, "M": M.tolist()}
        with open("calib.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved calib.json")
        break
    if k == 27:
        break

cv2.destroyAllWindows()
