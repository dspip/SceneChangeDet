from model.siameseNet.deeplab_v2 import  *
import cv2 
import numpy as np
import torch
from torch.autograd import Variable


def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

input_res = (640,480)

def single_layer_similar_heatmap_visual(output_t0,output_t1,dist_flag):

    interp = nn.Upsample(size=[input_res[0],input_res[1]], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return similar_distance_map_rz.data.cpu().squeeze(0).squeeze(0).numpy()

def RMS_Contrast(dist_map):

    h,w = dist_map.shape
    dist_map_l = np.resize(dist_map,(h*w))
    mean = np.mean(dist_map_l,axis=0)
    std = np.std(dist_map_l,axis=0,ddof=1)
    contrast = std / mean
    return contrast

model = SiameseNet() 
#model = deeplab_V2() 
#model = Deeplab_MS_Att_Scale()
#model = fcn32s() 
#pth = torch.load("model\\siameseNet\\deeplab_v2_voc12.pth",weights_only=True)
pth = torch.load("model\\siameseNet\\tsunami.pth",weights_only=True,map_location=torch.device("cpu"))
if "state_dict" in pth :
    ptsd = pth["state_dict"]
else:
    ptsd = pth

sd = model.state_dict()
assert(len(sd.keys()) == len(ptsd.keys()))

for sdk,ptk in zip(sd.keys(),ptsd.keys()):
    assert(sdk == ptk)
    print(sdk," | ",ptk,"match")

model.load_state_dict(ptsd)
model.eval()
dev = torch.device("cuda")
model.to(dev)


with torch.no_grad():
    #res = model(torch.zeros((1,3 ,640,480)),torch.ones((1, 3,640,480)))
    #print(res)

    video_path = 'rapik4.ts'

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    prev_frame = None
    counter = 0
    # Loop through video frames
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        
        # If frame is not read properly (end of video or error)
        if not ret:
            print("Reached end of video or error reading frame")
            break
        counter += 1

        if counter < 100:
            continue;

        frame = cv2.resize(frame,(input_res[0],input_res[1]))

        res = None
        if prev_frame is not None and frame is not None : 
            #f = torch.from_numpy(frame.reshape(3,1920,1080)[:,:640,:480]).unsqueeze(0).float()
            #pf = torch.from_numpy(prev_frame.reshape(3,1920,1080)[:,0:640,0:480]).unsqueeze(0).float()
            f = torch.from_numpy(frame.reshape(3,input_res[0],input_res[1])).unsqueeze(0).float().to(dev)
            pf = torch.from_numpy(prev_frame.reshape(3,input_res[0],input_res[1])).unsqueeze(0).float().to(dev)
            #print(f.shape,pf.shape)
            res = model(f,pf)
        if res :
            #for t in res:
            #    print(t[0].shape,t[1].shape)

            t0,t1,t2 = res
            print(len(res[2]))
            t0 = single_layer_similar_heatmap_visual(t0[0],t0[1],"l2")
            t1 = single_layer_similar_heatmap_visual(t1[0],t1[1],"l2")
            t2 = single_layer_similar_heatmap_visual(t2[0],t2[1],"l2")
            print(t2.shape)

            t0C = RMS_Contrast(t0)
            t1C = RMS_Contrast(t1)
            t2C = RMS_Contrast(t2)


            res = (t0C + t1C +t2)/2
            #print(res.shape)
            print(res.shape)
            cv2.imshow("mask",res);
        # Display the frame (you can add processing here)
        cv2.imshow('Video Playback', frame)
        prev_frame = frame
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

