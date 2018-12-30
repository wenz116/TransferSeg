function get_prior(video_name)

addpath(genpath('../utils/MBS/'));

base_dir = '../data/DAVIS/';
flow_dir  = [base_dir,  'Motion/480p/' video_name '/'];
prior_dir = [base_dir,  'Prior/480p/' video_name '/'];

cmd = ['mkdir -p ' prior_dir];
system(cmd);

image_names = dir([flow_dir '/*.png']);
num_img = length(image_names);

for i=1:num_img
	%disp(image_names(i).name);
    paramMBplus = getParam();

	I = imread([flow_dir image_names(i).name]);
    [pMap2] = doMBS(I, paramMBplus); 
	
	prior_path = [prior_dir image_names(i).name];
	imwrite(pMap2,prior_path);
end
