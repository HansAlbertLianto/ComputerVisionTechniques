%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       CZ4003 - COMPUTER VISION                        %
%                                LAB 1                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Point Processing + Spatial Filtering + Frequency Filtering +          %
% Imaging Geometry                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    by Hans Albert Lianto, U1620116K                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.1 CONTRAST SKETCHING                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get MRT train image
pic = imread('mrt-train.jpg');

% convert to grayscale image matrix format
bw_pic = rgb2gray(pic);

% display normal image
figure
imshow(bw_pic);

% contrast sketching
cntr_pic = imsubtract(bw_pic, double(min(bw_pic(:))));
cntr_pic = immultiply(cntr_pic, 255 ./ double(max(bw_pic(:)) - min(bw_pic(:))));

% display contrasted image
figure
imshow(cntr_pic);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.2 HISTOGRAM EQUALIZATION                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get image intensity histogram of picture with 10 bins
figure
imhist(bw_pic, 10);

% get image intensity histogram of picture with 256 bins
figure
imhist(bw_pic, 256);

% perform histogram equalization
hist_pic = histeq(bw_pic, 255);

% resdisplay image intensity histograms
figure
imhist(hist_pic, 10);

figure
imhist(hist_pic, 256);

% apply histogram equalization again
hist_pic_again = histeq(hist_pic, 255);

% redisplay histograms again
figure
imhist(hist_pic_again, 10);

figure
imhist(hist_pic_again, 255);

% apply histogram equalization yet again
hist_pic_again_again = histeq(hist_pic_again, 255);
 
% redisplay histograms yet again
figure
imhist(hist_pic_again_again, 10);
 
figure
imhist(hist_pic_again_again, 255);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.3 LINEAR SPATIAL FILTERING                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate filters
x_dimension = 5;
y_dimension = 5;
deviation_i = 1.0;
deviation_ii = 2.0;

a = x_dimension / 2;
b = y_dimension / 2;

% generate empty filters
filter_i = zeros([y_dimension, x_dimension]);
filter_ii = zeros([y_dimension, x_dimension]);

% iterate through empty filters and fill with Gaussian function
for y_index = 1:y_dimension
    for x_index = 1:x_dimension
        
        % get values for s and t
        s = y_index - ceil(y_dimension / 2);
        t = x_index - ceil(x_dimension / 2);
        
        filter_i(y_index, x_index) = exp(-(s.^2 + t.^ 2) ./ (2 .* (deviation_i) .^2)) ./ (2 .* pi .* (deviation_i).^2);
        filter_ii(y_index, x_index) = exp(-(s.^2 + t.^ 2) ./ (2 .* (deviation_ii) .^2)) ./ (2 .* pi .* (deviation_ii).^2);
    end
end

% normalize filters so sum of elements is 1
sum_filter_i = sum(filter_i, 'all');
sum_filter_ii = sum(filter_ii, 'all');

filter_i = filter_i ./ sum_filter_i;
filter_ii = filter_ii ./ sum_filter_ii;

% visualizing the 3D filters
x = -floor(x_dimension / 2) : 1 : floor(x_dimension / 2);
y = -floor(y_dimension / 2) : 1 : floor(y_dimension / 2);
[X, Y] = meshgrid(x, y);

figure
mesh(X, Y, filter_i)

figure
mesh(X, Y, filter_ii)

% apply filters to image
noisy_pic = imread('ntu-gn.jpg');

filtered_pic_i = uint8(conv2(noisy_pic, filter_i, 'same'));
filtered_pic_ii = uint8(conv2(noisy_pic, filter_ii, 'same'));

% display images
figure
imshow(noisy_pic);

figure
imshow(filtered_pic_i);

figure
imshow(filtered_pic_ii);

% get speckle noise image
speckle_img = imread('ntu-sp.jpg');

% apply filter to image
filtered_speckled_pic_i = uint8(conv2(speckle_img, filter_i, 'same'));
filtered_speckled_pic_ii = uint8(conv2(speckle_img, filter_ii, 'same'));

% display images
figure
imshow(speckle_img);

figure
imshow(filtered_speckled_pic_i);

figure
imshow(filtered_speckled_pic_ii);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.4 MEDIAN FILTERING                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% apply median filter to both images
median_filtered_noisy_pic_3 = medfilt2(noisy_pic, [3 3]);
median_filtered_noisy_pic_5 = medfilt2(noisy_pic, [5 5]);
median_filtered_speckled_pic_3 = medfilt2(speckle_img, [3 3]);
median_filtered_speckled_pic_5 = medfilt2(speckle_img, [5 5]);

% display images
figure
imshow(median_filtered_noisy_pic_3)

figure
imshow(median_filtered_noisy_pic_5)

figure
imshow(median_filtered_speckled_pic_3)

figure
imshow(median_filtered_speckled_pic_5)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.5 SUPPRESSING NOISE INTERFERENCE PATTERNS                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display picture with interference
interference_pic = imread('pck-int.jpg');
imshow(interference_pic)

% obtaining Fourier transform of image using fft2
interference_ft = fft2(interference_pic);

% estimate power spectrum of image
interference_ps = abs(interference_ft);

% display power spectrum
figure
imagesc(fftshift(interference_ps.^0.1));
colormap('default');

% display power spectrum without fftshift
figure
imagesc(interference_ps.^0.1);
colormap('default');

% obtain x, y coordinates of peaks via cursor click input
[x, y] = ginput(2);

x = round(x);
y = round(y);

% get Fourier transform matrix dimensions
[y_columns, x_rows] = size(interference_ft);

% set zero in 5x5 neighborhood of Fourier transform peaks
for x_range = -2:1:2
    for y_range = -2:1:2
        if (x + x_range > 0) & (x + x_range <= x_rows) & (y + y_range > 0) & (y + y_range <= y_columns)
            interference_ft(y + y_range, x + x_range) = 0;
        end
    end
end

% recompute power spectrum
new_interference_ps = abs(interference_ft);

% display power spectrum
figure
imagesc(fftshift(new_interference_ps.^0.1));
colormap('default');

% convert changed Fourier transform back to image
processed_interference_pic = uint8(abs(ifft2(interference_ft)));

% display resultant image
figure
imshow(processed_interference_pic)

% set zero in whole lines
for x_range = -1:1:1
    for y_range = -1:1:1
        if (x + x_range > 0) & (x + x_range <= x_rows) & (y + y_range > 0) & (y + y_range <= y_columns)
            interference_ft(y + y_range, :) = 0;
            interference_ft(:, x + x_range) = 0;
        end
    end
end

% recompute power spectrum
improved_interference_ps = abs(interference_ft);

% display power spectrum
figure
imagesc(fftshift(improved_interference_ps.^0.1));
colormap('default');

% convert changed Fourier transform back to image
improved_processed_interference_pic = uint8(abs(ifft2(interference_ft)));

% display resultant image
figure
imshow(improved_processed_interference_pic)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get primate cage image
interference_pic = rgb2gray(imread('primate-caged.jpg'));
figure
imshow(interference_pic)

% obtaining Fourier transform of image using fft2
interference_ft = fft2(interference_pic);

% estimate power spectrum of image
interference_ps = abs(interference_ft);

% display power spectrum
figure
imagesc(fftshift(interference_ps.^0.1));
colormap('default');

% display power spectrum without fftshift
figure
imagesc(interference_ps.^0.01);
colormap('default');

% obtain x, y coordinates of peaks via cursor click input
[x, y] = ginput;

x = round(x);
y = round(y);

% get Fourier transform matrix dimensions
[y_columns, x_rows] = size(interference_ft);

% set zero in 3x3 neighborhood of Fourier transform peaks
for x_range = -1:1:1
    for y_range = -1:1:1
        if (x + x_range > 0) & (x + x_range <= x_rows) & (y + y_range > 0) & (y + y_range <= y_columns)
            interference_ft(y + y_range, x + x_range) = 0;
        end
    end
end

% recompute power spectrum
new_interference_ps = abs(interference_ft);

% display power spectrum
figure
imagesc(fftshift(new_interference_ps.^0.1));
colormap('default');

% convert changed Fourier transform back to image
processed_interference_pic = uint8(abs(ifft2(interference_ft)));

% display resultant image
figure
imshow(processed_interference_pic)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.6 UNDOING PERSPECTIVE DISTORTION OF PLANAR SURFACE                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display slanted book image
slant_image = imread('book.jpg');
figure
imshow(slant_image)

% get four edges of the book
[X Y] = ginput(4);

% declare four edges of the book, so order is known
%
% ORDER: top left, top right, bottom right, bottom left
book_edges_x = [1;210;210;1];
book_edges_y = [1;1;297;297];

% set up matrices
A = zeros([8, 8]);
u = zeros([8, 1]);

% setup v
for index = 1 : length(v)
   if mod(index, 2) == 0
       v(index) = book_edges_y(ceil(index / 2));
   else
       v(index) = book_edges_x(ceil(index / 2));
   end
end

% setup A
for row = 1 : size(A, 1)
    for col = 1 : size(A, 2)
        if (col >= 7)
            if mod(col, 2) == 0
                if mod(row, 2) == 0
                    A(row, col) = -book_edges_y(ceil(row / 2)) .* Y(ceil(row / 2));
                else
                    A(row, col) = -book_edges_x(ceil(row / 2)) .* Y(ceil(row / 2));
                end
            else
                if mod(row, 2) == 0
                    A(row, col) = -book_edges_y(ceil(row / 2)) .* X(ceil(row / 2));
                else
                    A(row, col) = -book_edges_x(ceil(row / 2)) .* X(ceil(row / 2));
                end
            end
        elseif mod(row, 2) == 0
            if (col == 4)
                A(row, col) = X(ceil(row / 2));
            elseif (col == 5)
                A(row, col) = Y(ceil(row / 2));
            elseif (col == 6)
                A(row, col) = 1;
            else
                A(row, col) = 0;
            end
        else
            if (col == 1)
                A(row, col) = X(ceil(row / 2));
            elseif (col == 2)
                A(row, col) = Y(ceil(row / 2));
            elseif (col == 3)
                A(row, col) = 1;
            else
                A(row, col) = 0;
            end
        end
    end
end

% get projective transformation parameter vector
u = A \ v;

% get transformation matrix
U = reshape([u;1], 3, 3)';

% get transform
transform = maketform('projective', U');

% create aligned image
aligned_image = imtransform(slant_image, transform, 'XData', [0 210], 'YData', [0 297]);

% display result image
figure
imshow(aligned_image)

