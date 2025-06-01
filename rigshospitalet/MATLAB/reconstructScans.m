mainFolder = 'jeppes_project/data/scans';

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

function [nii] = processToHU(img_processed, scanType, method, geo)
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

    % Create NIfTI file
    nii = make_nii(img_HU, geo.dVoxel);
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
    scanTypeFolders = {'Head', 'Thorax', 'Pelvis', 'Pelvis Large', 'Short Thorax', 'Thorax Advanced'}; 
    found = false;
    for scanType = scanTypeFolders
        if exist(fullfile('../../../data/reconstructions/OSSARTFull', scanType{1}, [subFolders(i).name '.nii']), 'file')
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
        
        % Create sampling indices for all sparse reconstructions ahead of time
        %keepIdx_every2nd = 1:2:totalProjs;
        keepIdx_every3rd = 1:3:totalProjs;
        %keepIdx_every5th = 1:5:totalProjs;
        keepIdx_every10th = 1:10:totalProjs;
        
        % Define sampling patterns as a struct for easy iteration
        samplings = struct();
        samplings.full = struct('name', 'Full', 'indices', 1:totalProjs, 'angles', angles);
        %samplings.p50 = struct('name', '50pct', 'indices', keepIdx_every2nd, 'angles', angles(keepIdx_every2nd));
        samplings.p33 = struct('name', '33pct', 'indices', keepIdx_every3rd, 'angles', angles(keepIdx_every3rd));
        %samplings.p20 = struct('name', '20pct', 'indices', keepIdx_every5th, 'angles', angles(keepIdx_every5th));
        samplings.p10 = struct('name', '10pct', 'indices', keepIdx_every10th, 'angles', angles(keepIdx_every10th));
        
        % Define reconstruction methods
        %methods = {'FDK', 'MLEM', 'OSASDPOCS', 'OSSART'};
        methods = {'FDK','OSSART'};
        
        % Process all combinations of methods and sampling patterns
        for methodIdx = 1:length(methods)
            method = methods{methodIdx};
            samplingNames = fieldnames(samplings);
            
            for s = 1:length(samplingNames)
                samplingName = samplingNames{s};
                sampling = samplings.(samplingName);
                
                fprintf('Starting %s %s reconstruction...\n', method, sampling.name);
                
                projFiltered = proj(:, :, sampling.indices);
                anglesFiltered = sampling.angles;
                
                % Perform reconstruction
                if strcmp(method, 'FDK')
                    img = FDK(projFiltered, geo, anglesFiltered, 'gpuids', gpuids);
                    img = im2DDenoise(img, 'median', 3);
                end
            
                if strcmp(method, 'MLEM')
                    img = MLEM(projFiltered, geo, anglesFiltered, 50, 'gpuids', gpuids);
                end

                if strcmp(method, 'OSASDPOCS')
                    img = OS_ASD_POCS(projFiltered, geo, anglesFiltered, 25, 'gpuids', gpuids);
                end

                if strcmp(method, 'OSSART')
                    img = OS_SART(projFiltered, geo, anglesFiltered, 35, 'gpuids', gpuids);
                end
                
                
                % Process the reconstructed image
                imgReoriented = processOrientation(img);
                niiHU = processToHU(imgReoriented, scanType, method, geo);
                
                % Determine folder path for HU image
                folderPath = fullfile('../../../data/reconstructions', [method sampling.name], scanType);
                if ~exist(folderPath, 'dir')
                    mkdir(folderPath);
                end
                
                % Save HU version
                save_nii(niiHU, fullfile(folderPath, [subFolders(i).name '.nii']));
                
                % Save non-HU version only for full reconstructions
                % if strcmp(sampling.name, 'Full')
                %     folderPathNonHU = fullfile('../../../data/reconstructions', [method 'NotHU'], scanType);
                %     if ~exist(folderPathNonHU, 'dir')
                %         mkdir(folderPathNonHU);
                %     end
     
                %     img_clipped = imclip(imgReoriented, [0, 0.1]);
                %     nii_nonHU = make_nii(img_clipped, geo.dVoxel);
                %     save_nii(nii_nonHU, fullfile(folderPathNonHU, [subFolders(i).name '.nii']));
                %     %plotImg(img_clipped, 'step', 1, 'Dim', 'z', 'savegif', fullfile(folderPathNonHU, [subFolders(i).name '.gif']), 'colormap', 'gray');
                %     clear img_clipped nii_nonHU;
                % end
                
                % Clean up
                clear img imgHU nii projFiltered anglesFiltered imgReoriented niiHU;
            end
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