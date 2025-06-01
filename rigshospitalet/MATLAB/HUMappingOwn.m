function HU3D = HUMappingOwn(img3D, Protocol)
% Pixel value to HU value Mapping
% Method: using CATPhan 604 calibration data
% Input:
%       img3D: reconstructed image matrix in pixel value
%       Protocol: CBCT scan protocol
% Output:
%       HU3D: image matrix in CT HU value
% Date: 2021-09-03
% Author: Yi Du, yi.du@hotmail.com

%Remove space in protocol
Protocol = strrep(Protocol, ' ', '');

HU3D = zeros(numel(img3D), 1);
jsonFile = 'jeppes_project/Thesis/rigshospitalet/MATLAB/ownPhantomPixel2HU.json';
demoJsonFile = 'jeppes_project/Thesis/rigshospitalet/MATLAB/demo_Pixel2HU.json';

if ~isfile(jsonFile)
    warning('Using DEMO pixel2HU transform. We recommend creating your own and storing it in TIGRE/Common/data/pixel2HU.json');
    jsonData = fileread(demoJsonFile);
else
    jsonData = fileread(jsonFile);
end

% Decode JSON
Pixel2HU = jsondecode(jsonData);

% Extract calibration data for the given protocol
if isfield(Pixel2HU, Protocol)
    tmp = Pixel2HU.(Protocol);
else
    error('Protocol not found in Pixel2HU JSON file.');
end

[xData, yData] = prepareCurveData(tmp(1, :), tmp(2, :));

% 2nd-order Polynomial Fitting: Set up fittype and options.
ft = fittype('poly2');

% Fit model to data.
[fitresult, gof] = fit(xData, yData, ft);

HU3D = reshape(fitresult(img3D), size(img3D));

end
