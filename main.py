from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def run():
    model = YOLO('yolov8n-seg.yaml')
    model.train(data='./datasets/data.yaml', epochs=1000, device=0, imgsz=640)


def predict():
    model = YOLO('runs/segment/train8/weights/best.pt')
    model.predict(source='test0919_kmeans', save=True, show=True, save_txt=True, device=0)
    # results = model.predict(source=frame, save=False, save_txt=False, device=0, stream=True, classes=0)


def distance_y():
    model = YOLO('runs/segment/train8/weights/best.pt')
    image = cv2.imread('./test_image/IMG_7298.JPG')
    result = model.predict(source=image, save=False, show=False, stream=True, save_txt=False, device=0)

    for r in result:
        find = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # r = r.cpu()
        # r = r.numpy()

        pos = r.masks.xy[0]

        find = cv2.fillPoly(find, [pos.astype(np.int32)], (255))

        plt.bar([i for i in range(image.shape[0])], find.sum(axis=1) / 255)

        cv2.imshow("find", find)
        cv2.imshow("img", image)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def welding_fix_scale():
    model = YOLO('runs/segment/train8/weights/best.pt')
    image = cv2.imread('./test0919/test7.jpg')
    result = model.predict(source=image, save=False, show=False, stream=True, save_txt=False, device=0)

    # Kmeans
    src = image.copy()

    data = src.reshape((-1, 3)).astype(np.float32)

    K = 5
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5,
                                      cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape(src.shape)

    # find line
    for r in result:
        find = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # r = r.cpu()
        # r = r.numpy()

        box = r.boxes.xyxy[0].cpu().numpy()
        pos = r.masks.xy[0]

        print(box)

        mask = cv2.fillPoly(mask, [pos.astype(np.int32)], (255))
        find = cv2.polylines(find, [pos.astype(np.int32)], True, (255))

        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        canny = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 21)
        # canny = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        # canny = cv2.Canny(gray, 200, 250)

        image = cv2.polylines(image, [pos.astype(np.int32)], True, (255, 0, 0))

        height = int((box[3] - box[1]) / 15)
        width = height

        windows_pos_x = box[0]
        windows_pos_y = box[1]

        plot_x = []
        plot_y = []

        while windows_pos_y + height < box[3]:
            while windows_pos_x + width < box[2]:
                current_window = canny[int(windows_pos_y):int(windows_pos_y+height), int(windows_pos_x):int(windows_pos_x+width)]
                hist = cv2.calcHist([current_window], [0], None, [256], [0, 255])

                plot_x.append(windows_pos_x)
                sum = hist[220:256].sum()
                plot_y.append(sum)

                print(f'current pos : w={windows_pos_x}, h={windows_pos_y}')
                print(f'sum = {sum}')

                # cv2.imshow('curr', current_window)
                # cv2.waitKey(0)
                windows_pos_x = windows_pos_x + 1

            plt.bar(plot_x, plot_y)
            plt.show()

            cv2.imshow('current', canny[int(windows_pos_y):int(windows_pos_y+height), int(box[0]):int(box[2])])
            cv2.waitKey(0)

            plot_x = []
            plot_y = []
            windows_pos_x = box[0]
            windows_pos_y = windows_pos_y + height

        cv2.imshow("find", find[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        cv2.imshow("img", image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        cv2.imshow("canny", canny[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        cv2.imshow("and", cv2.bitwise_or(find[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                                         canny[int(box[1]):int(box[3]), int(box[0]):int(box[2])]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def extract_area():
    model = YOLO('runs/segment/train8/weights/best.pt')

    path_dir = "C:\\Users\\user\\Desktop\\Repository\\Welding_Yolo\\more_test00"
    file_list = os.listdir(path_dir)

    for file in file_list:
        print(path_dir + '\\' + file)
        image = cv2.imread(path_dir + '\\' + file)
        result = model.predict(source=image, save=False, show=False, stream=True, save_txt=False, device=0)

        for r in result:
            find = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            # r = r.cpu()
            # r = r.numpy()

            # print(r.masks)
            pos = r.masks.xy[0]
            # print(pos[0])
            find = cv2.fillPoly(find, [pos.astype(np.int32)], (255))
            cv2.imwrite(path_dir + '_area' + '\\' + file, find)

def kmeans():
    path_dir = "C:\\Users\\user\\Desktop\\Repository\\Welding_Yolo\\test0919"
    file_list = os.listdir(path_dir)

    for file in file_list:
        print(path_dir + '\\' + file)
        src = cv2.imread(path_dir + '\\' + file)

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        data = src.reshape((-1, 3)).astype(np.float32)

        K = 5
        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5,
                                          cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        dst = res.reshape(src.shape)

        cv2.imwrite(path_dir + '_kmeans' + '\\' + file, dst)


if __name__ == '__main__':
    # run()
    # predict()
    # distance_y()
    welding_fix_scale()
    # extract_area()
    # kmeans()
