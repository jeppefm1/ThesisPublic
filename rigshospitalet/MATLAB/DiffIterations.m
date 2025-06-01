mainFolder = 'jeppes_project/data/scans';

function [mae, rmse, ssim_val, psnr_val] = computeDifferences(img, img_ref)
    % Ensure images are of the same type
    img = single(img);
    img_ref = single(img_ref);

    % Rescale images to [0,1]
    img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    img_ref = (img_ref - min(img_ref(:))) / (max(img_ref(:)) - min(img_ref(:)));

    % Compute differences
    diff = img - img_ref;

    % Mean Absolute Error (MAE)
    mae = mean(abs(diff(:)));

    % Root Mean Square Error (RMSE)
    rmse = sqrt(mean(diff(:).^2));

    % Structural Similarity Index (SSIM)
    ssim_val = ssim(img, img_ref);

    % Peak Signal-to-Noise Ratio (PSNR)
    maxI = max(img_ref(:));
    mse = mean(diff(:).^2);
    psnr_val = 20 * log10(maxI / sqrt(mse));
end


function img_clipped = imclip(img, range)
    img_clipped = max(min(img, range(2)), range(1));
end

function [img_processed] = processOrientation(img)
    % Apply standard image transformations required for all reconstructions
    img_processed = permute(img, [2 1 3]);
    %img_processed = flip(img_processed, 1);  %Radiological view flip.
    img_processed = flip(img_processed, 2); 
    img_processed = flip(img_processed, 3);
end

function [img_HU] = processToHU(img_processed, scanType, method)
    % Map method and scanType to protocol
    if strcmp(method, 'FDK')
        prefix = 'FDK';
    elseif strcmp(method, 'MLEM')
        prefix = 'MLEM50';
    elseif strcmp(method, 'OSASDPOCS')
        prefix = 'MLEM50';
    elseif strcmp(method, 'OSSART')
        prefix = 'OSSART50';
    else
        error('Unknown method: %s', method);
    end

    switch scanType
        case 'Head'
            protocol = [prefix, 'Head'];
        case 'Pelvis'
            protocol = [prefix, 'Pelvis'];
        case 'Thorax'
            protocol = [prefix, 'Pelvis']; % Same mapping as Pelvis
        case 'Short Thorax'
            protocol = [prefix, 'Pelvis']; % Same mapping as Pelvis
        case 'Thorax Advanced'
            protocol = [prefix, 'Pelvis']; % Same mapping as Pelvis
        case 'Pelvis Large'
            protocol = [prefix, 'PelvisLarge'];
        otherwise
            error('Unknown scan type: %s', scanType);
    end

    % Apply HU mapping and clipping
    img_HU = HUMappingOwn(img_processed, protocol);
    img_HU = imclip(img_HU, [-1000, 2000]);
end


% Get all subfolders
subFolders = dir(mainFolder);
subFolders = subFolders([subFolders.isdir]);  % Keep only directories
subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'}));

% Sort alphabetically by folder name
[~, idx] = sort({subFolders.name});
subFolders = subFolders(idx);
numScans = length(subFolders);

% Get list of available GPUs
gpuids = GpuIds();

% Loop through each subfolder
for i = 1:numScans
    % Reset GPU at beginning of each scan
    reset(gpuDevice);
    
    currentSubFolder = fullfile(mainFolder, subFolders(i).name);
    fprintf('Processing Scan %d/%d: %s\n', i, numScans, subFolders(i).name);
    
    % Check if the scan has already been reconstructed
    scanTypeFolders = {'Head', 'Thorax', 'Pelvis', 'Pelvis Large'}; 
    found = false;
    for scanType = scanTypeFolders
        if exist(fullfile('../../../data/reconstructions/OSSART75', scanType{1}, [subFolders(i).name '.nii']), 'file')
            fprintf('Scan %s already reconstructed in %s. Skipping...\n', subFolders(i).name, scanType{1});
            found = true;
            break;
        end
    end
    if found
        continue;
    end
    
    try
        fprintf('Loading data...\n');
        [proj, geo, angles, scanType] = VarianDataLoaderOwn(currentSubFolder, 'acdc', true, 'sc', true, 'bh', true, 'gpuids', gpuids);
        
        % Convert to single precision to save memory
        proj = single(proj);
        totalProjs = size(proj, 3);
        fprintf('Data loaded: %d total projections\n', totalProjs);

        
        % Define sampling patterns as a struct for easy iteration
        samplings = struct();
        samplings.full = struct('name', 'Full', 'indices', 1:totalProjs, 'angles', angles);
        
          
        % Define iteration counts
        iterCounts = [10, 20, 35, 50,75];

        % Always compute OSSART75 first
        fprintf('Starting OSSART75 reconstruction...\n');
        projFiltered = proj(:, :, samplings.full.indices);
        anglesFiltered = samplings.full.angles;
        tic;
        OSSART75 = OS_SART(proj, geo, angles, 75, 'gpuids', gpuids);
        reconstruction75_time = toc; % Elapsed time in seconds

        % Process the OSSART75 reconstruction
        imgReoriented = processOrientation(OSSART75);
        HUOSSART75 = processToHU(imgReoriented, scanType, 'OSSART');
        % Create NIfTI file
        niiHUOSSART75 = make_nii(HUOSSART75, geo.dVoxel);
        folderPath = fullfile('../../../data/reconstructions', 'OSSART75', scanType);
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
        save_nii(niiHUOSSART75, fullfile(folderPath, [subFolders(i).name '.nii']));

        metricsFile = fullfile(folderPath, [subFolders(i).name '_metrics.txt']);
        fid = fopen(metricsFile, 'w');
        fprintf(fid, 'Time: %.3f\n', reconstruction75_time);
        fclose(fid);


        % Compute other reconstructions and compare to OSSART75
        for iter = iterCounts(1:end-1)  % Exclude 75, already done
            method = ['OSSART' num2str(iter)];
            fprintf('Starting %s reconstruction...\n', method);

            projFiltered = proj(:, :, samplings.full.indices);
            anglesFiltered = samplings.full.angles;

            tic;
            img = OS_SART(projFiltered, geo, anglesFiltered, iter, 'gpuids', gpuids);
            reconstruction_time = toc; % Elapsed time in seconds

            % Process the reconstructed image
            imgReoriented = processOrientation(img);
            HU = processToHU(imgReoriented, scanType, 'OSSART');
            niiHU = make_nii(HU, geo.dVoxel);
            
            
            % Save the image
            folderPath = fullfile('../../../data/reconstructions', method, scanType);
            if ~exist(folderPath, 'dir')
                mkdir(folderPath);
            end
            save_nii(niiHU, fullfile(folderPath, [subFolders(i).name '.nii']));

            % Compute Differences with OSSART75
            fprintf('Computing error metrics between %s and OSSART75...\n', method);
            [mae, rmse, ssim_val, psnr_val] = computeDifferences(HU, HUOSSART75);

            % Save metrics
            metricsFile = fullfile(folderPath, [subFolders(i).name '_metrics.txt']);
            fid = fopen(metricsFile, 'w');
            fprintf(fid, 'MAE: %.3f\nRMSE: %.3f\nSSIM: %.3f\nPSNR: %.3f\nTime: %.3f\n', mae, rmse, ssim_val, psnr_val, reconstruction_time);
            fclose(fid);

            % Clean up
            clear img imgReoriented niiHU;
        end

        
        % Free up RAM - explicitly gather GPU arrays if needed
        if exist('proj', 'var') && isa(proj, 'gpuArray')
            proj = gather(proj);
        end
        if exist('geo', 'var') && isa(geo, 'gpuArray')
            geo = gather(geo);
        end
        if exist('angles', 'var') && isa(angles, 'gpuArray')
            angles = gather(angles);
        end
        
    catch ME
        fprintf('Error processing scan %s: %s\n', subFolders(i).name, ME.message);
        disp(getReport(ME, 'extended'));
        
        % Ensure memory is released even if error occurs
        clearvars -except i mainFolder subFolders gpuids numScans min_val max_val;
        java.lang.Runtime.getRuntime.gc;
        reset(gpuDevice);
        continue;
    end
    
    % Clear all variables at end of loop
    clearvars -except i mainFolder subFolders gpuids numScans min_val max_val;
    java.lang.Runtime.getRuntime.gc;
    reset(gpuDevice);
    
    % Force MATLAB to perform garbage collection
    pause(2);  % Give system a moment to clear memory
    
end
fprintf('All scans processed.\n');