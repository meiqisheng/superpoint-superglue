import cv2
import numpy as np
import secrets

###################################################### 1. 定义SIFT类 ####################################################
class  CSift:
	def __init__(self,num_octave,num_scale,sigma):
		self.sigma = sigma	#初始尺度因子
		self.num_scale = num_scale #层数
		self.num_octave = 3 #组数，后续重新计算
		self.contrast_t = 0.04#弱响应阈值
		self.eigenvalue_r = 10#hessian矩阵特征值的比值阈值
		self.scale_factor = 1.5#求取方位信息时的尺度系数
		self.radius_factor = 3#3被采样率
		self.num_bins = 36 #计算极值点方向时的方位个数
		self.peak_ratio = 0.8 #求取方位信息时，辅方向的幅度系数

#################################################### 6. 匹配 ############################################################
def do_match(img_src1,kp1,des1,img_src2,kp2,des2,embed=1,pt_flag=0,MIN_MATCH_COUNT = 10):
    ## 1. 对关键点进行匹配 ##
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1, des2 = np.array(des1).astype(np.float32), np.array(des2).astype(np.float32)#需要转成array
    matches = flann.knnMatch(des1, des2, k=2)  # matches为list，每个list元素由2个DMatch类型变量组成,分别是最邻近和次邻近点

    good_match = []
    for m in matches:
        if m[0].distance < 0.7 * m[1].distance:  # 如果最邻近和次邻近的距离差距较大,则认可
            good_match.append(m[0])
    ## 2. 将2张图画在同一张图上 ##
    img1 = img_src1.copy()
    img2 = img_src2.copy()
    h1, w1 = img1.shape[0],img1.shape[1]
    h2, w2 = img2.shape[0],img2.shape[1]
    new_w = np.max([w1, w2])
    new_h = h1 + h2
    new_img =  np.zeros((new_h, new_w,3), np.uint8) if len(img_src1.shape)==3 else  np.zeros((new_h, new_w), np.uint8)
    w_offset1 = int(0.5 * (new_w - w1))
    w_offset2 = int(0.5 * (new_w - w2))
    if len(img_src1.shape) == 3:
        new_img[:h1,w_offset1:w_offset1 + w2,:] = img1  # 左边画img1
        new_img[h1:h1 + h2,w_offset2:w_offset2 + w2,:] = img2  # 右边画img2
    else:
        new_img[:h1,w_offset1:w_offset1 + w2] = img1  # 左边画img1
        new_img[h1:h1 + h2,w_offset2:w_offset2 + w2] = img2  # 右边画img2
    ##3. 两幅图存在足够的匹配点，两幅图匹配成功，将匹配成功的关键点进行连线 ##
    if len(good_match) > MIN_MATCH_COUNT:
        src_pts = []
        dst_pts = []
        mag_err_arr=[]
        angle_err_arr=[]
        for m in good_match:
            if pt_flag==0:#point是百分比
                src_pts.append([kp1[m.queryIdx].pt[0] * img1.shape[1], kp1[m.queryIdx].pt[1] * img1.shape[0]])#保存匹配成功的原图关键点位置
                dst_pts.append([kp2[m.trainIdx].pt[0] * img2.shape[1], kp2[m.trainIdx].pt[1] * img2.shape[0]])#保存匹配成功的目标图关键点位置
            else:
                src_pts.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])  # 保存匹配成功的原图关键点位置
                dst_pts.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])  # 保存匹配成功的目标图关键点位置

            mag_err = np.abs(kp1[m.queryIdx].response - kp2[m.trainIdx].response) / np.abs(kp1[m.queryIdx].response )
            angle_err = np.abs(kp1[m.queryIdx].angle - kp2[m.trainIdx].angle)
            mag_err_arr.append(mag_err)
            angle_err_arr.append(angle_err)

        if embed!=0 :#若图像2是图像1内嵌入另一个大的背景中，则在图像2中，突出显示图像1的边界
            M = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, 5.0)[0]  # 根据src和dst关键点，寻求变换矩阵
            src_w, src_h = img1.shape[1], img1.shape[0]
            src_rect = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]]).reshape(-1, 1, 2).astype(
                np.float32)  # 原始图像的边界框
            # dst_rect = cv2.perspectiveTransform(src_rect, M)  # 经映射后，得到dst的边界框
            # img2 = cv2.polylines(img2, [np.int32(dst_rect)], True, 255, 3, cv2.LINE_AA)  # 将边界框画在dst图像上，突出显示
            if len(new_img.shape) == 3:
                new_img[h1:h1+h2, w_offset2:w_offset2 + w2,:] = img2  # 右边画img2
            else:
                new_img[h1:h1+h2, w_offset2:w_offset2 + w2] = img2  # 右边画img2

        new_img = new_img if len(new_img.shape) == 3 else  cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
        flag = 0
        for pt1, pt2 in zip(src_pts, dst_pts):
            if(flag == 16):
                flag = 0
                bgr=np.random.randint(0,255,3,dtype=np.int32)
                cv2.line(new_img, tuple(np.int32(np.array(pt1) + [w_offset1, 0])),tuple(np.int32(np.array(pt2) + [w_offset2, h2])), color=(np.int(bgr[0]),np.int(bgr[1]),np.int(bgr[2])))
            flag += 1
    return new_img, M


def get_numOfOctave(img):
	num = round (np.log(min(img.shape[0],img.shape[1]))/np.log(2) )-1
	return num


if __name__ == '__main__':
    MIN_MATCH_COUNT = 10
    sift = CSift(num_octave=4,num_scale=3,sigma=1.6)
    img_src1 = cv2.imread('assets/temple.jpeg',-1)
    #img_src1 = cv2.resize(img_src1, (0, 0), fx=.25, fy=.25)
    img_src2 = cv2.imread('assets/origin.jpeg', -1)
    #img_src2 = cv2.resize(img_src2, (0, 0), fx=.5, fy=.5)
    # 2. 使用opencv自带sift算子
    sift.num_octave = get_numOfOctave(img_src1)
    opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=int(sift.num_octave),
                                    contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
    kp1 = opencv_sift.detect(img_src1)
    kp1,des1 = opencv_sift.compute(img_src1,kp1)

    sift.num_octave = get_numOfOctave(img_src2)
    opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=int(sift.num_octave),
                                    contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
    kp2 = opencv_sift.detect(img_src2)
    kp2, des2 = opencv_sift.compute(img_src2, kp2)
    pt_flag = 1

    # 3. 做匹配
    reu_img, M = do_match(img_src1, kp1, des1, img_src2, kp2, des2, embed=1, pt_flag=pt_flag,MIN_MATCH_COUNT=3)

    TestPt = np.float32([[272,208], [272, 268], [297, 267], [296, 222], [353, 183], [353, 245], [380, 239], [384, 152]])
    TestPt2 = list()
    reu_img = cv2.fillPoly(reu_img, [np.array(TestPt,dtype=np.int32)], 255)

    pts = np.float32(TestPt).reshape(-1,2).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)#.reshape(1, -1).reshape(-1,2)
    dst = np.int32(dst)
    for point in dst:
        point[0][1] += img_src1.shape[0]
        TestPt2.append(point)
    reu_img = cv2.fillPoly(reu_img, [np.array(TestPt2,dtype=np.int32)], 255)
    cv2.imshow('sift',reu_img)
    cv2.imwrite('sift.jpg',reu_img)