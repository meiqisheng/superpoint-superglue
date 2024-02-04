
# import cv2
# import os
# import onnxruntime as ort
# import numpy as np
# import datetime
# import time

# def sample(descriptor, keypoint, s=8):

#    n, c, h, w = descriptor.shape

#    x, y = keypoint[0], keypoint[1]
#    new_x = (x -8/2 +0.5) / (w*s - s/2 - 0.5) * (w-1)
#    new_y = (y -8/2 +0.5) / (h*s - s/2 - 0.5) * (h-1)

#    floor_x, ceil_x = int(np.floor(new_x)), int(np.ceil(new_x))
#    floor_y, ceil_y = int(np.floor(new_y)), int(np.ceil(new_y))

#    top_left  = descriptor[:, :, floor_y, floor_x]
#    top_right = descriptor[:, :, floor_y, ceil_x]
#    bottom_left  = descriptor[:, :, ceil_y, floor_x]
#    bottom_right = descriptor[:, :, ceil_y, ceil_x]

#    weight_top_left  = (ceil_x-new_x) * (ceil_y-new_y)
#    weight_top_right = (new_x-floor_x) * (ceil_y-new_y)
#    weight_bottom_left  = (ceil_x-new_x) * (new_y-floor_y)
#    weight_bottom_right = (new_x-floor_x) * (new_y-floor_y)

#    res_descriptor = top_left*weight_top_left + top_right*weight_top_right + bottom_left*weight_bottom_left + bottom_right*weight_bottom_right

#    normalize = max(np.sqrt(np.sum(np.power(res_descriptor, 2))), 1e-12)

#    return res_descriptor / normalize

# class Matching(object):

#     def __init__(self):
#         current_path = os.path.dirname(os.path.abspath(__file__))
#         print('当前路径: ', current_path)

#         self.superpoint_sess = ort.InferenceSession(os.path.join(current_path, 'weights/superpoint.onnx'))
#         # self.superglue_sess = ort.InferenceSession(os.path.join(current_path, 'superglue.onnx'))

#     def run(self, template, image):
#         template_keypoints, template_score = self.superpoint_sess.run(None, {'input': template})
#         template_score = np.expand_dims(template_score, axis=(0,1))
#         template_keypoints = np.expand_dims(np.array(template_keypoints, dtype=np.int32), axis=0)
#         # template_descriptors = [sample(template_descriptor, template_keypoints[0,k]) for k in range(template_keypoints.shape[1])]
#         # template_descriptors = np.array(template_descriptors, dtype=np.float32)
#         # template_descriptors = np.transpose(template_descriptors, (1,2,0))
#         template_keypoint = np.array(template_keypoints, dtype=np.float32)

#         image_keypoints, image_score = self.superpoint_sess.run(None, {'input': image})
#         image_score = np.expand_dims(image_score, axis=(0,1))
#         image_keypoints = np.expand_dims(np.array(image_keypoints, dtype=np.int32), axis=0)
#         # image_descriptors = [sample(image_descriptor, image_keypoints[0,k]) for k in range(image_keypoints.shape[1])]
#         # image_descriptors = np.array(image_descriptors, dtype=np.float32)
#         # image_descriptors = np.transpose(image_descriptors, (1,2,0))
#         image_keypoint = np.array(image_keypoints, dtype=np.float32)
#         # print('template_score:{}\n template_descriptor:{}, '.format(template_score, template_descriptor))

#         # match_output = self.superglue_sess.run(None, {
#         #         'image0': template,
#         #         'keypoints0': template_keypoint,
#         #         'scores0': template_score,
#         #         'descriptors0': template_descriptors,
#         #         'image1': image,
#         #         'keypoints1': image_keypoint,
#         #         'scores1': image_score,
#         #         'descriptors1': image_descriptors,
#         #        })
#         # print('template_score:{}\n image_score:{}, '.format(template_score, image_score))
#         # return template_keypoint, image_keypoint, match_output[0], match_output[2]
#         return template_keypoint, image_keypoint


# if __name__ == '__main__':


#     temp_pic = cv2.cvtColor(cv2.imread('assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg'), cv2.COLOR_BGR2GRAY)
#     temp_pic = cv2.resize(temp_pic, (960, 540))
#     template = np.array(temp_pic, dtype=np.float32)/255.
#     template = np.expand_dims(template, axis=(0,1))

#     curr_pic = cv2.cvtColor(cv2.imread('assets//phototourism_sample_images/london_bridge_49190386_5209386933.jpg'), cv2.COLOR_BGR2GRAY)
#     curr_pic = cv2.resize(curr_pic, (960, 540))
#     image = np.array(curr_pic, dtype=np.float32)/255.
#     image = np.expand_dims(image, axis=(0,1))

#     # template_keypoint, image_keypoint, match_output0, match_output2 = Matching().run(template, image)
#     template_keypoint, image_keypoint = Matching().run(template, image)

#     pts00 = np.squeeze(template_keypoint)
#     pts11 = np.squeeze(image_keypoint)
#     # print('template_keypoints:{}\n image_keypoint:{}, '.format(pts00, pts11))
#     # print('pts0.shape:{}\n pts1.shape:{}'.format(pts00.shape, pts11.shape))

#     height, width = temp_pic.shape
#     img12 = np.concatenate([temp_pic, curr_pic], axis=1)
#     img12 = cv2.cvtColor(img12, cv2.COLOR_GRAY2BGR)

#     for point, point1 in zip(pts00, pts11):
#         bgr=np.random.randint(0,255,3,dtype=np.int32)#随机颜色
#         color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
#         # print('point:{}, point1:{}'.format(point, point))
#         cv2.circle(img12, (int(point[0]), int(point[1])), 4, color, -1)
#         cv2.circle(img12, (int(point1[0] + width), int(point1[1])), 4, color, -1)
#         cv2.line(img12, [int(point[0]), int(point[1])], [int(point1[0] + width), int(point1[1])], color, 2)
    
#     cv2.imwrite('match-result.jpg', img12)


#     #frame_tensor = torch.from_numpy(temp_pic/255.).float()[None, None].to('cpu')
#     #print(frame_tensor.shape)
#     #data = {'image': frame_tensor}




import cv2
import os
import onnxruntime as ort
import numpy as np
import datetime
import time

def sample(descriptor, keypoint, s=8):

   n, c, h, w = descriptor.shape

   x, y = keypoint[0], keypoint[1]
   new_x = (x -8/2 +0.5) / (w*s - s/2 - 0.5) * (w-1)
   new_y = (y -8/2 +0.5) / (h*s - s/2 - 0.5) * (h-1)

   floor_x, ceil_x = int(np.floor(new_x)), int(np.ceil(new_x))
   floor_y, ceil_y = int(np.floor(new_y)), int(np.ceil(new_y))

   top_left  = descriptor[:, :, floor_y, floor_x]
   top_right = descriptor[:, :, floor_y, ceil_x]
   bottom_left  = descriptor[:, :, ceil_y, floor_x]
   bottom_right = descriptor[:, :, ceil_y, ceil_x]

   weight_top_left  = (ceil_x-new_x) * (ceil_y-new_y)
   weight_top_right = (new_x-floor_x) * (ceil_y-new_y)
   weight_bottom_left  = (ceil_x-new_x) * (new_y-floor_y)
   weight_bottom_right = (new_x-floor_x) * (new_y-floor_y)

   res_descriptor = top_left*weight_top_left + top_right*weight_top_right + bottom_left*weight_bottom_left + bottom_right*weight_bottom_right

   normalize = max(np.sqrt(np.sum(np.power(res_descriptor, 2))), 1e-12)

   return res_descriptor / normalize

class Matching(object):

    def __init__(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        print('当前文件夹： ', current_path)

        self.superpoint_sess = ort.InferenceSession(os.path.join(current_path, 'weights/point_robot_true.onnx'))
        self.superglue_sess = ort.InferenceSession(os.path.join(current_path, 'weights/superglue_robot_true.onnx'))

    def run(self, template, image):
        template_keypoints, template_score, template_descriptor = self.superpoint_sess.run(None, {'input': template})
        template_score = np.expand_dims(template_score, axis=(0,1))
        template_keypoints = np.expand_dims(np.array(template_keypoints, dtype=np.int32), axis=0)
        template_descriptors = [sample(template_descriptor, template_keypoints[0,k]) for k in range(template_keypoints.shape[1])]
        template_descriptors = np.array(template_descriptors, dtype=np.float32)
        template_descriptors = np.transpose(template_descriptors, (1,2,0))
        template_keypoint = np.array(template_keypoints, dtype=np.float32)

        image_keypoints, image_score, image_descriptor = self.superpoint_sess.run(None, {'input': image})
        image_score = np.expand_dims(image_score, axis=(0,1))
        image_keypoints = np.expand_dims(np.array(image_keypoints, dtype=np.int32), axis=0)
        image_descriptors = [sample(image_descriptor, image_keypoints[0,k]) for k in range(image_keypoints.shape[1])]
        image_descriptors = np.array(image_descriptors, dtype=np.float32)
        image_descriptors = np.transpose(image_descriptors, (1,2,0))
        image_keypoint = np.array(image_keypoints, dtype=np.float32)
        # print('template_score:{}\n template_descriptor:{}, '.format(template_score, template_descriptor))

        # match_output = self.superglue_sess.run(None, {
        #         'image0': template,
        #         'keypoints0': template_keypoint,
        #         'scores0': template_score,
        #         'descriptors0': template_descriptors,
        #         'image1': image,
        #         'keypoints1': image_keypoint,
        #         'scores1': image_score,
        #         'descriptors1': image_descriptors,
        #        })
        # # print('template_score:{}\n image_score:{}, '.format(template_score, image_score))
        # return template_keypoint, image_keypoint, match_output[0], match_output[2]
        
        for _ in range(1):
            match_output = self.superglue_sess.run(None, {
                'image0': template,
                'keypoints0': template_keypoint,
                'scores0': template_score,
                'descriptors0': template_descriptors,
                'image1': image,
                'keypoints1': image_keypoint,
                'scores1': image_score,
                'descriptors1': image_descriptors,
                })

        matches = np.squeeze(match_output[0])
        confidence = np.squeeze(match_output[2])

        kpts0 = np.squeeze(template_keypoints)
        kpts1 = np.squeeze(image_keypoints)

        # valid good matches
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        match_confidence = confidence[valid]
        pts0 = mkpts0.reshape(-1,1,2)
        pts1 = mkpts1.reshape(-1,1,2)

        # import pdb; pdb.set_trace()
        print(matches[valid])
        print(len(matches[valid]))

        #_M, mask = cv2.findHomography(np.array(pts0, dtype=np.int32), np.array(pts1, dtype=np.int32), cv2.RANSAC, 5.0)
        _M, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        print('matrix', _M)
        
        return template_keypoint, image_keypoint


if __name__ == '__main__':


    temp_pic = cv2.cvtColor(cv2.imread('assets/temple.jpeg'), cv2.COLOR_BGR2GRAY)
    mask = cv2.imread('mask1.png', 0)
    temp_pic = cv2.resize(temp_pic, (640, 512))
    template = np.array(temp_pic, dtype=np.float32)/255.
    template = np.expand_dims(template, axis=(0,1))

    curr_pic = cv2.cvtColor(cv2.imread('assets/origin.jpeg'), cv2.COLOR_BGR2GRAY)
    curr_pic = cv2.resize(curr_pic, (640, 512))
    image = np.array(curr_pic, dtype=np.float32)/255.
    image = np.expand_dims(image, axis=(0,1))

    template_keypoint, image_keypoint = Matching().run(template, image)
    

    pts00 = np.squeeze(template_keypoint)
    pts11 = np.squeeze(image_keypoint)
    # print('template_keypoints:{}\n image_keypoint:{}, '.format(pts00, pts11))
    # print('pts0.shape:{}\n pts1.shape:{}'.format(pts00.shape, pts11.shape))

    TestPt = np.float32([[272,208], [272, 268], [297, 267], [296, 222], [353, 183], [353, 245], [380, 239], [384, 152]])
    TestPt2 = list()
    height, width = temp_pic.shape
    img12 = np.concatenate([temp_pic, curr_pic], axis=0)
    img12 = cv2.cvtColor(img12, cv2.COLOR_GRAY2BGR)

    P1 = list()
    P2 = list()
    imshowi = 0

    for point, point1 in zip(pts00, pts11):
        bgr=np.random.randint(0,255,3,dtype=np.int32)#随机颜色
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        # print('point:{}, point1:{}'.format(point, point))
        if(imshowi == 16):
            imshowi = 0
            cv2.circle(img12, (int(point[0]), int(point[1])), 3, color, -1)
            cv2.circle(img12, (int(point1[0]), int(point1[1]) + height), 3, color, -1)
            cv2.line(img12, [int(point[0]), int(point[1])], [int(point1[0]), int(point1[1] + height)], color, 1)
        P1.append(point)
        P2.append(point1)
        imshowi += 1
    
    P1=np.array(P1)
    P2=np.array(P2)
    _M, mask = cv2.findHomography(P1, P2, cv2.RANSAC, 5.0)

    img12 = cv2.fillPoly(img12, [np.array(TestPt,dtype=np.int32)], 255)
    print(TestPt)
    pts = np.float32(TestPt).reshape(-1,2).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, _M)#.reshape(1, -1).reshape(-1,2)
    dst = np.int32(dst)
    for point in dst:
        point[0][1] += height
        TestPt2.append(point)
    img12 = cv2.fillPoly(img12, [np.array(TestPt2,dtype=np.int32)], 255)

    print('matrix', _M)

    cv2.imwrite('match-result.jpg', img12)


    # template_keypoint, image_keypoint, match_output0, match_output2 = Matching().run(template, image)

    # pts00 = np.squeeze(template_keypoint)
    # pts11 = np.squeeze(image_keypoint)
    # # print('template_keypoints:{}\n image_keypoint:{}, '.format(pts00, pts11))
    # # print('pts0.shape:{}\n pts1.shape:{}'.format(pts00.shape, pts11.shape))

    # height, width = temp_pic.shape
    # img12 = np.concatenate([temp_pic, curr_pic], axis=1)
    # img12 = cv2.cvtColor(img12, cv2.COLOR_GRAY2BGR)

    # for point, point1 in zip(pts00, pts11):
    #     bgr=np.random.randint(0,255,3,dtype=np.int32)#随机颜色
    #     color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    #     # print('point:{}, point1:{}'.format(point, point))
    #     cv2.circle(img12, (int(point[0]), int(point[1])), 4, color, -1)
    #     cv2.circle(img12, (int(point1[0] + width), int(point1[1])), 4, color, -1)
    #     cv2.line(img12, [int(point[0]), int(point[1])], [int(point1[0] + width), int(point1[1])], color, 2)
    
    # cv2.imwrite('match-result.jpg', img12)


    #frame_tensor = torch.from_numpy(temp_pic/255.).float()[None, None].to('cpu')
    #print(frame_tensor.shape)
    #data = {'image': frame_tensor}



