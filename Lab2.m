%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       CZ4003 - COMPUTER VISION                        %
%                                LAB 2                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Edge Detection + Hough Transform + Pixel Sum-of-squares Difference +  %
% Bag-of-words Method with Spatial Pyramid Matching (SPM)               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    by Hans Albert Lianto, U1620116K                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set preferences
iptsetpref('ImshowAxesVisible', 'off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.1 EDGE DETECTION                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display image in grayscale
running_img = rgb2gray(imread('macritchie.jpg'));

figure
imshow(running_img)

% setup sobel filters
h_sobel_filter = [-1, -2, -1;
                 0, 0, 0;
                 1, 2, 1];

v_sobel_filter = [-1, 0, 1;
                 -2, 0, 2;
                 -1, 0, 1];
             
% apply filters to image
h_sobel_img = conv2(running_img, h_sobel_filter, 'same');
v_sobel_img = conv2(running_img, v_sobel_filter, 'same');

% display filtered images
figure
imshow(uint8(h_sobel_img))

figure
imshow(uint8(v_sobel_img))

% get combined image and display it
sobel_img = h_sobel_img .^ 2 + v_sobel_img .^ 2;

% display combined image
figure
imshow(uint8(sobel_img))

% thresholding the image
sobel_threshold_img_10000 = uint8(sobel_img > 10000) .* 255;
sobel_threshold_img_20000 = uint8(sobel_img > 20000) .* 255;
sobel_threshold_img_30000 = uint8(sobel_img > 30000) .* 255;
sobel_threshold_img_40000 = uint8(sobel_img > 40000) .* 255;
sobel_threshold_img_50000 = uint8(sobel_img > 50000) .* 255;
sobel_threshold_img_60000 = uint8(sobel_img > 60000) .* 255;
sobel_threshold_img_70000 = uint8(sobel_img > 70000) .* 255;
sobel_threshold_img_80000 = uint8(sobel_img > 80000) .* 255;
sobel_threshold_img_90000 = uint8(sobel_img > 90000) .* 255;
sobel_threshold_img_100000 = uint8(sobel_img > 100000) .* 255;
sobel_threshold_img_110000 = uint8(sobel_img > 110000) .* 255;
sobel_threshold_img_120000 = uint8(sobel_img > 120000) .* 255;

% display thresholded images
figure
imshow(sobel_threshold_img_10000)

figure
imshow(sobel_threshold_img_20000)

figure
imshow(sobel_threshold_img_30000)

figure
imshow(sobel_threshold_img_40000)

figure
imshow(sobel_threshold_img_50000)

figure
imshow(sobel_threshold_img_60000)

figure
imshow(sobel_threshold_img_70000)

figure
imshow(sobel_threshold_img_80000)

figure
imshow(sobel_threshold_img_90000)

figure
imshow(sobel_threshold_img_100000)

figure
imshow(sobel_threshold_img_110000)

figure
imshow(sobel_threshold_img_120000)

% recompute binary border image using Canny edge detection
tl = [0, 0.02, 0.04, 0.06, 0.08];
th = 0.1;
sigma = [1, 2, 3, 4, 5];

canny_img_sigma_1 = edge(running_img, 'canny', [tl(3) th], sigma(1));
canny_img_sigma_2 = edge(running_img, 'canny', [tl(3) th], sigma(2)); 
canny_img_sigma_3 = edge(running_img, 'canny', [tl(3) th], sigma(3)); 
canny_img_sigma_4 = edge(running_img, 'canny', [tl(3) th], sigma(4)); 
canny_img_sigma_5 = edge(running_img, 'canny', [tl(3) th], sigma(5)); 

canny_img_tl_1 = edge(running_img, 'canny', [tl(1) th], sigma(2));
canny_img_tl_2 = edge(running_img, 'canny', [tl(2) th], sigma(2));
canny_img_tl_3 = edge(running_img, 'canny', [tl(4) th], sigma(2));
canny_img_tl_4 = edge(running_img, 'canny', [tl(5) th], sigma(2));

% display all images
figure
imshow(canny_img_sigma_1)

figure
imshow(canny_img_sigma_2)

figure
imshow(canny_img_sigma_3)

figure
imshow(canny_img_sigma_4)

figure
imshow(canny_img_sigma_5)

figure
imshow(canny_img_tl_1)

figure
imshow(canny_img_tl_2)

figure
imshow(canny_img_tl_3)

figure
imshow(canny_img_tl_4)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.2 LINE FINDING USING HOUGH TRANSFORM                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% apply the Radon transform
[H, xp] = radon(canny_img_sigma_1);

% set x-axis
theta = 0:179;

% set Radon transform display
figure
iptsetpref('ImshowAxesVisible', 'on');
imshow(H, [], 'Xdata', theta, 'Ydata', xp, 'InitialMagnification', 'fit');
xlabel('\theta (degrees)')
ylabel('x''')
colormap(gca, hot), colorbar

% get x' and theta such that H is highest
[radius_max, theta_max] = ind2sub(size(H), find(H == max(H(:))));
radius_max = xp(radius_max);
theta_max = theta(theta_max);

% obtain line parameters A and B in normal line equation from
% obtained x' and theta
[A, B] = pol2cart(theta_max * pi / 180, radius_max);
B = -B;

% finding C
C = A * (A + size(combined_image, 2) / 2) + B * (B + size(combined_image, 1) / 2);

% project line onto image
xl = 1;
xr = size(combined_image, 2);
yl = (C - A * xl) / B;
yr = (C - A * xr) / B;

% reset preferences
iptsetpref('ImshowAxesVisible', 'off');

% display image with line obtained by Radon transform
figure
imshow(running_img)
line([xl xr], [yl yr])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.3 PIXEL INTENSITY SUM-OF-SQUARES DIFFERENCE (SSD) AND 3D STEREO     %
%     VISION                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display two stereo images
stereo_img_l = rgb2gray(imread('corridorl.jpg'));
stereo_img_r = rgb2gray(imread('corridorr.jpg'));

figure
imshow(stereo_img_l)

figure
imshow(stereo_img_r)

% get disparity map of left image
map = disparity_map(stereo_img_l, stereo_img_r, 11, 11);

% display disparity map and ideal disparity map
figure
imshow(map, [-15 15])

ideal_stereo_img = imread('corridor_disp.jpg');
figure
imshow(ideal_stereo_img)

% display two stereo images
triclops_img_l = rgb2gray(imread('triclops-i2l.jpg'));
triclops_img_r = rgb2gray(imread('triclops-i2r.jpg'));

figure
imshow(triclops_img_l)

figure
imshow(triclops_img_r)

% get disparity map of left image
map = disparity_map(triclops_img_l, triclops_img_r, 11, 11);

% display disparity map
figure
imshow(map, [-15 15])

ideal_stereo_img = imread('triclops-id.jpg');
figure
imshow(ideal_stereo_img)

% get disparity map of left image
map = disparity_map(triclops_img_l, triclops_img_r, 21, 21);

% display disparity map
figure
imshow(map, [-15 15])

