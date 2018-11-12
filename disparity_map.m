%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.3 PIXEL INTENSITY SUM-OF-SQUARES DIFFERENCE (SSD) AND 3D STEREO     %
%     VISION                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to compute for disparity map for two images, with no of rows %
% and columns in neighbour template matrix as extra two parameters      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function disp_map = disparity_map(first_img, second_img, template_rows, template_cols)

    % make sure image matrices are being passed to first to first two
    % arguments
    if (~ismatrix(first_img) || ~ismatrix(second_img))
        error('ERROR: An image matrix should be passed to first two arguments');
    end

    % make sure template dimensions are odd
    if (mod(template_rows, 2) == 0 || mod(template_cols, 2) == 0)
        error('ERROR: Template matrix must have odd number dimensions');
    end
    
    % initialize disparity map
    disp_map = ones(size(first_img, 1) - template_rows + 1, size(first_img, 2) - template_cols + 1);

    % get half of size of template matrix
    hsize_rows = floor(template_rows / 2);
    hsize_cols = floor(template_cols / 2);
    
    % perform SSD matching on each pixel on the first image
    for img_row = 1 + hsize_rows : size(first_img, 1) - hsize_rows
        
        % disp(img_row)
        
        for img_col = 1 + hsize_cols : size(first_img, 2) - hsize_cols
            
            % set min SSD
            min_ssd = inf;
            min_coor = img_col;
            
            % get neighbouring template matrix of pixel
            temp_first_img = double(first_img(img_row - hsize_rows : img_row + hsize_rows, img_col - hsize_cols : img_col + hsize_cols));
            
            % constrain search to neighbouring 14 (15 - 1) pixels only
            % so disparity is less than 15
            for disparity_coord = max(1 + hsize_cols, img_col - 14) : img_col
                
                % get neighbour template of 2nd image
                temp_second_img = double(second_img(img_row - hsize_rows : img_row + hsize_rows, disparity_coord - hsize_cols : disparity_coord + hsize_cols));

                % compute parts of SSD
                ssd_1 = temp_second_img .^ 2;
                ssd_2 = temp_second_img .* temp_first_img;

                ssd = sum(ssd_1, 'all') - 2 * sum(ssd_2, 'all');
                
                % update minimum SSD if SSD is lower than the current
                % minimum SSD
                if (ssd < min_ssd)
                    min_ssd = ssd;
                    min_coor = disparity_coord;
                end
            end
            
            % Update the disparity map
            disp_map(img_row - hsize_rows, img_col - hsize_cols) = img_col - min_coor;
            
        end
    end
    