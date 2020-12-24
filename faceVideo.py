import cv2

# 调用摄像头摄像头
cap = cv2.VideoCapture(0)

# 检测人脸
def detect_face(sample_image):
    # OpenCV 人脸检测
    face_patterns_people = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces_people = face_patterns_people.detectMultiScale(sample_image, scaleFactor=1.15, minNeighbors=5)
    return faces_people


# 圣诞帽
hats = []
for i in range(4):
    hats.append(cv2.imread('img/hat%d.png' % i, -1))


# 显示人脸轮廓
def showFace(faces, sample_image):
    for x, y, w, h in faces:
        cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return sample_image


# 戴帽子
def inHat(faces, sample_image):
    for face in faces:
        # 随机一顶帽子
        # hat = random.choice(hats)
        hat = hats[1]
        # 调整帽子尺寸
        scale = face[3] / hat.shape[0] * 1.2
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
        # 根据人脸坐标调整帽子位置
        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2)
        y_offset = int(face[1] - hat.shape[0] / 2)
        # 计算贴图位置，注意防止超出边界的情况
        x1, x2 = max(x_offset, 0), min(x_offset + hat.shape[1], sample_image.shape[1])
        y1, y2 = max(y_offset, 0), min(y_offset + hat.shape[0], sample_image.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1
        # 透明部分的处理
        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h
        # 按3个通道合并图片
        for c in range(0, 3):
            sample_image[y1:y2, x1:x2, c] = (
                        alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * sample_image[y1:y2, x1:x2, c])


# 调用摄像头摄像头
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces_people = detect_face(frame)
    # showFace(faces_people,frame)
    inHat(faces_people, frame)
    # 展示图片
    cv2.imshow("input", frame)
    # 监听用户案件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
