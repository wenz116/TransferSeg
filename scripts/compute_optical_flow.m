function compute_optical_flow(video_name)

addpath(genpath('../utils/OpticalFlow/'));

base_dir = '../data/DAVIS/';
image_dir = [base_dir,  'JPEGImages/480p/' video_name '/'];
flow_dir  = [base_dir,  'Motion/480p/' video_name '/'];

cmd = ['mkdir -p ' flow_dir];
system(cmd);

image_names = dir([image_dir '/*.jpg']);
num_img = length(image_names);

for i=1:num_img
	%disp(image_names(i).name);
	if i==num_img
		img1 = imread([image_dir image_names(i).name]);
		img2 = imread([image_dir image_names(i-1).name]);
	else
		img1 = imread([image_dir image_names(i).name]);
		img2 = imread([image_dir image_names(i+1).name]);
	end
	
	[vx,vy,warpI2]=get_optical_flow(img1,img2);
	flow(:,:,1)=vx;
	flow(:,:,2)=vy;
	flow_img = flowToColor(flow);
	
	flow_path = [flow_dir image_names(i).name(1:end-3) 'png'];
	imwrite(flow_img,flow_path);
end
