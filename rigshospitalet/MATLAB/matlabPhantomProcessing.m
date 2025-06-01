function [img_processed] = processOrientation(img)
    % Apply standard image transformations required for all reconstructions
    img_processed = permute(img, [2 1 3]);
    %img_processed = flip(img_processed, 1);%Radiological view flip.
    img_processed = flip(img_processed, 2); 
    img_processed = flip(img_processed, 3);
end

gpuids = GpuIds();


% Rawdata Loading and Conditioning
[proj, geo, angles, scanType] = VarianDataLoaderOwn('jeppes_project/data/Catphan phantom/thoraxpelvis2025-03-25_145012/38ddbcfd-30da-492d-89e0-56f0b3ad2908', 'acdc', true, 'sc', true, 'bh', true, 'gpuids', gpuids);
% FDK Reconstruction
% imgFDK = FDK(proj, geo, angles, 'gpuids', gpuids);
% imgFDK = processOrientation(imgFDK);
% imgFDK = HUMappingOwn(imgFDK, "FDKPelvis");

% nii = make_nii(imgFDK, geo.dVoxel);
% save_nii(nii, 'imgFDKPhamtomThoraxPelvisHU.nii');

%Maximum likelihood reconctrustion
% imgMLEM = MLEM(proj, geo, angles, 50, 'gpuids', gpuids);
% imgMLEM = processOrientation(imgMLEM);
% imgMLEM = HUMappingOwn(imgMLEM, "MLEM50Pelvis");

% niiMLEM = make_nii(imgMLEM, geo.dVoxel);
% save_nii(niiMLEM, 'imgMLEMPhamtomThoraxPelvisHU.nii');

%Ossart
imgOSSART = OS_SART(proj, geo, angles, 50, 'gpuids', gpuids);
imgOSSART = processOrientation(imgOSSART);
%imgMLEM = HUMappingOwn(imgMLEM, "MLEM50Pelvis");

niiOSSART = make_nii(imgOSSART, geo.dVoxel);
save_nii(niiOSSART, 'imgOSSARTPhamtomThoraxPelvis.nii');



[proj, geo, angles, scanType] = VarianDataLoaderOwn('jeppes_project/data/Catphan phantom/pelvisL2025-03-25_145155/1a8498c0-eb7c-4c33-bb69-a75532b0f3d0', 'acdc', true, 'sc', true, 'bh', true, 'gpuids', gpuids);
% % FDK Reconstruction
% imgFDK = FDK(proj, geo, angles, 'gpuids', gpuids);
% imgFDK = processOrientation(imgFDK);
% imgFDK = HUMappingOwn(imgFDK, "FDKPelvisLarge");

% nii = make_nii(imgFDK, geo.dVoxel);
% save_nii(nii, 'imgFDKPhamtomPelvisLargeHU.nii');

% %Maximum likelihood reconctrustion
% imgMLEM = MLEM(proj, geo, angles, 50, 'gpuids', gpuids);
% imgMLEM = processOrientation(imgMLEM);
% imgMLEM = HUMappingOwn(imgMLEM, "MLEM50PelvisLarge");

% niiMLEM = make_nii(imgMLEM, geo.dVoxel);
% save_nii(niiMLEM, 'imgMLEMPhamtomPelvisLargeHU.nii');

%Ossart
imgOSSART = OS_SART(proj, geo, angles, 50, 'gpuids', gpuids);
imgOSSART = processOrientation(imgOSSART);

niiOSSART = make_nii(imgOSSART, geo.dVoxel);
save_nii(niiOSSART, 'imgOSSARTPhamtomPelvisLarge.nii');



[proj, geo, angles, scanType] = VarianDataLoaderOwn('jeppes_project/data/Catphan phantom/head2025-03-25_144743/c67b02f7-9503-4a02-a72c-5aca3568072f', 'acdc', true, 'sc', true, 'bh', true, 'gpuids', gpuids);
% FDK Reconstruction
% imgFDK = FDK(proj, geo, angles, 'gpuids', gpuids);
% imgFDK = processOrientation(imgFDK);
% imgFDK = HUMappingOwn(imgFDK, "FDKHead");

% nii = make_nii(imgFDK, geo.dVoxel);
% save_nii(nii, 'imgFDKPhamtomHeadHU.nii');

% %Maximum likelihood reconctrustion
% imgMLEM = MLEM(proj, geo, angles, 50, 'gpuids', gpuids);
% imgMLEM = processOrientation(imgMLEM);
% imgMLEM = HUMappingOwn(imgMLEM, "MLEM50Head");

% niiMLEM = make_nii(imgMLEM, geo.dVoxel);
% save_nii(niiMLEM, 'imgMLEMPhamtomHeadHU.nii');

%Ossart
imgOSSART = OS_SART(proj, geo, angles, 50, 'gpuids', gpuids);
imgOSSART = processOrientation(imgOSSART);

niiOSSART = make_nii(imgOSSART, geo.dVoxel);
save_nii(niiOSSART, 'imgOSSARTPhamtomHead.nii');



% [proj, geo, angles, scanType] = VarianDataLoaderOwn('jeppes_project/data/Phantom/child2025-03-25_144639/2c91a398-4170-4375-9cc6-2830a1e64e37', 'acdc', true, 'sc', true, 'bh', true, 'gpuids', gpuids);
% % FDK Reconstruction
% imgFDK = FDK(proj, geo, angles, 'gpuids', gpuids);
% imgFDK = processOrientation(imgFDK);
% imgFDK = HUMappingOwn(imgFDK, "FDKChild");

% nii = make_nii(imgFDK, geo.dVoxel);
% save_nii(nii, 'imgFDKPhamtomChildHU.nii');

% %Maximum likelihood reconctrustion
% imgMLEM = MLEM(proj, geo, angles, 50, 'gpuids', gpuids);
% imgMLEM = processOrientation(imgMLEM);
% imgMLEM = HUMappingOwn(imgMLEM, "MLEM50Child");

% niiMLEM = make_nii(imgMLEM, geo.dVoxel);
% save_nii(niiMLEM, 'imgMLEMPhamtomChildHU.nii');
