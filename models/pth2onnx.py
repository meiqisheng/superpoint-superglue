import torch
from superpoint import SuperPoint
import os
import cv2

default_config = {
    'descriptor_dim': 256,
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': -1,
    'remove_borders': 4,
}
device ='cpu'

dummy_input = torch.zeros((1,)+(1,540,960), dtype=torch.float32)

model_test = SuperPoint(default_config).to(device)
print('当前路径：', os.getcwd())
model_statedict = torch.load("models/weights/superpoint_v1.pth")   #导入Gpu训练模型，导入为cpu格式
model_test.load_state_dict(model_statedict)  #将参数放入model_test中
model_test.eval()  # 测试，看是否报错
#下面开始转模型，cpu格式下
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_pic = cv2.cvtColor(cv2.imread('assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg'), cv2.COLOR_BGR2GRAY)
temp_pic = cv2.resize(temp_pic, (960, 540))
frame_tensor = torch.from_numpy(temp_pic/255.).float()[None, None].to('cpu')
print(frame_tensor.shape)
data = {'image': frame_tensor}

# input_names = ["input"]
# output_names = ["output"]
# output_names_list = ['keypoint', 'scores','descriptors']

# torch.onnx.export(model_test,
#                    dummy_input, 
#                    "models/weights/superpoint_v1.onnx", 
#                    opset_version=16, 
#                    verbose=False, 
#                   input_names= input_names,
#                   output_names= output_names_list,
#                    dynamic_axes= {"input": {2 : 'in_width', 3: 'int_height'}}
#                   )
torch.onnx.export(model_test, 
                  frame_tensor, 
                  "models/weights/superpoint_v1_self.onnx", 
                  verbose=True, 
                  input_names=['input'], 
                  output_names=['keypoints', 'scores', 'descriptors'], 
                  dynamic_axes={'input':{2:'hight',3:'width'}, 
                                'keypoints':{0:'point_size'}, 
                                'scores':{0:'point_size'}, 
                                'descriptors':{2:'hight',3:'width'}}, 
                                opset_version=16)



output = model_test(frame_tensor)
# import pdb; pdb.set_trace()

#    for k in data:
#        if isinstance(data[k], (list, tuple)):
#print(torch.stack(output['keypoints'][0]).shape)

keypoints = output['keypoints'][0].cpu().detach().numpy()
scores = output['scores'][0].cpu().detach().numpy()
descriptors = output['descriptors'][0].cpu().detach().numpy()
# descriptors_upsample = output['descriptors_upsample'][0].cpu().detach().numpy()
# import pdb; pdb.set_trace()

kc, ks = keypoints.shape
# stack_descriptors = [descriptors_upsample[:,keypoints[k,0],keypoints[k,1]] for k in range(kc)]
