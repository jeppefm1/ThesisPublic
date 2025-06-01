#Translated Python version of https://github.com/CERN/TIGRE/tree/aabf889e00a6aaee8d17fa1fd644d600077b69a9/MATLAB/Utilities/IO/VarianCBCT
#AI utilized


import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata, RectBivariateSpline
from multiprocessing import Pool, cpu_count


from scipy.ndimage import zoom
from scipy.signal import fftconvolve
import os
from glob import glob
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Any, Optional, Union
from pylinac.core.image import XIM
from multiprocessing import Pool
from tqdm import tqdm
import xmltodict

import tigre
import tigre.algorithms.single_pass_algorithms as algs
from tigre.utilities.geometry import Geometry
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter




# Process a single projection (moved outside the main function)
def process_projection(args):
    """
    Process a single projection for scatter correction
    
    Parameters:
    -----------
    args : tuple
        (idx, proj_slice, parameters)
    
    Returns:
    --------
    tuple
        (idx, corrected_projection)
    """
    idx, proj_slice, params = args
    
    # Unpack parameters
    s_blk, s_air_norm, airnorm, proj_shape, vs, us, dvs, dus, step_du, step_dv, sc_calib, ngroup, asg, gform, fft_gform_asg, gamma, mm2cm, niter, lambda_relax = params
    
    # Rescale blank scan
    cf = s_air_norm / airnorm[idx]
    s_blk_rescaled = (s_blk / cf).reshape(len(vs), len(us))
    
    # Interpolate blank scan to downsampled grid (once per projection)
    blk_interp = RectBivariateSpline(vs, us, s_blk_rescaled)
    blk_ds = blk_interp(dvs, dus)
    blk_ds[blk_ds < 0] = 0
    
    # Interpolate projection to downsampled grid (once per projection)
    proj_interp = RectBivariateSpline(vs, us, proj_slice.reshape(len(vs), len(us)))
    page = proj_interp(dvs, dus)
    page[page < 0] = 0
    
    # Initialize scatter estimate
    Is = np.zeros_like(page)
    
    # Pre-allocate buffers for iterative process
    term1 = np.zeros((page.shape[0], page.shape[1], ngroup))
    
    # Iterative Correction
    for j in range(niter):
        # Save previous scatter map
        Is_prv = Is.copy()
        
        # Estimate thickness (mm)
        thickness = sc_thickness_estimator(blk_ds, page, sc_calib, step_du, step_dv)
        
        # Smooth thickness
        thickness = sc_smooth_thickness(thickness, sc_calib, step_du, step_dv)
        
        # Group-based masks (vectorized)
        nmask = sc_group_mask_vec(thickness, ngroup, 
                                  np.array([float(sc_calib['CalibrationResults']['ObjectScatterModels']['ObjectScatterModel'][i]['Thickness']) 
                                          for i in range(ngroup)]))
        
        # Edge response function
        edgewt = sc_edge_response(thickness)
        
        # Group-based amplitude factors
        cfactor = sc_amplitude_factor(blk_ds, page, edgewt, sc_calib)
        
        # mm -> cm conversion
        thickness_cm = thickness * mm2cm
        
        # n-group summation (vectorized)
        for k in range(ngroup):
            term1[:,:,k] = page * nmask[:,:,k] * cfactor[:,:,k]
        
        # Calculate FFT components (optimized)
        tmp1 = np.zeros_like(page, dtype=complex)
        tmp2 = np.zeros_like(page, dtype=complex)
        
        for k in range(ngroup):
            fft_term1 = np.fft.fft2(term1[:,:,k])
            tmp1 += fft_term1 * fft_gform_asg[:,:,k]
            tmp2 += np.fft.fft2(thickness_cm * term1[:,:,k]) * fft_gform_asg[:,:,k]
        
        comp1 = np.real(np.fft.ifft2(tmp1))
        comp2 = np.real(np.fft.ifft2(tmp2))
        
        # fASKS scatter correction
        Is = (1 - gamma * thickness_cm) * comp1 + gamma * comp2
        
        # Update primary estimate
        page = page + lambda_relax * (Is_prv - Is)
        page[page < 0] = np.finfo(float).eps
    
    # Upsampling scatter map
    scmap_interp = RectBivariateSpline(dvs, dus, Is)
    scmap = scmap_interp(vs, us)
    
    # Handle extrapolation errors
    scmap[np.isnan(scmap)] = np.finfo(float).eps
    scmap[scmap < 0] = np.finfo(float).eps
    
    # Calculate scatter fraction
    sf = scmap / proj_slice
    
    # Handle special cases
    sf[~np.isfinite(sf)] = np.nan
    sf[sf > 1000] = np.nan
    
    # Fill NaN values
    sf = inpaint_nans(sf)
    
    # Median filtering
    sf = ndimage.median_filter(sf, size=(3, 3))
    
    # Scatter fraction cutoff
    sf = np.minimum(sf, 0.95)
    
    # Primary signal
    return idx, proj_slice * (1 - sf)


def scatter_correction(sc_calib, blk, blk_air_norm, proj, airnorm, geo):
    """
    Scatter Correction Module based on Adaptive Scatter Kernel Superposition
    
    Parameters:
    -----------
    sc_calib : dict
        Scatter calibration structure
    blk : ndarray
        Blank scan
    blk_air_norm : ndarray
        Blank scan air normalization
    proj : ndarray
        Projections with shape (y, x, angles)
    airnorm : ndarray
        Air normalization values
    geo : dict
        Geometry structure
    
    Returns:
    --------
    prim : ndarray
        Primary signal after scatter correction
    """
    # Center Coordinates (unit: mm)
    offset = geo.offDetector
    
    # Grid unit: mm
    us = (np.arange(-geo.nDetector[1]/2 + 0.5, geo.nDetector[1]/2, 1) * 
          geo.dDetector[1] - offset[1])
    vs = (np.arange(-geo.nDetector[0]/2 + 0.5, geo.nDetector[0]/2, 1) * 
          geo.dDetector[0] - offset[0])
    
    # Downsampling factor
    ds_rate = 12
    
    # Downsampled grid
    dus = us[::ds_rate]
    dvs = vs[::ds_rate]
    
    step_du = np.mean(np.diff(dus))
    step_dv = np.mean(np.diff(dvs))
    
    # Pre-compute grids (unit: mm)
    dugd, dvgd = np.meshgrid(dus, dvs)  # downsampled detector
    
    # Blank scan
    s_blk = np.sum(blk, axis=2)
    s_air_norm = np.sum(blk_air_norm)
    
    # n-thickness group number and boundaries
    obj_scatter_models = sc_calib['CalibrationResults']['ObjectScatterModels']['ObjectScatterModel']
    ngroup = len(obj_scatter_models)
    
    # Anti-scatter grid kernel (precomputed)
    asg = sc_asg_kernel(sc_calib, geo, dus, dvs)
    
    # Component Weights: gamma (gamma = 0 for SKS)
    gamma = float(obj_scatter_models[0]['ObjectScatterFit']['gamma'])
    
    # unit conversion: mm -> cm
    mm2cm = 1/10
    
    # Iteration parameters
    niter = 8
    lambda_relax = 0.005  # relaxation factor
    
    # Primary signal matrix
    prim = np.zeros_like(proj)
    
    # Pre-compute form functions (only depends on geometry)
    gform = sc_form_func(sc_calib, dugd, dvgd)
    
    # Pre-compute FFT of form functions * ASG
    fft_gform_asg = np.zeros((dugd.shape[0], dugd.shape[1], ngroup), dtype=complex)
    for k in range(ngroup):
        fft_gform_asg[:,:,k] = np.fft.fft2(gform[:,:,k] * asg)
    
    # Prepare parallel processing
    n_jobs = min(cpu_count(), 8)  # Use up to 8 cores
    
    # Pack parameters
    params = (s_blk, s_air_norm, airnorm, proj.shape, vs, us, dvs, dus, 
              step_du, step_dv, sc_calib, ngroup, asg, gform, 
              fft_gform_asg, gamma, mm2cm, niter, lambda_relax)
    
    # Create task list
    tasks = [(i, proj[:,:,i], params) for i in range(proj.shape[2])]
    
    # Process all projections with progress updates
    print("Processing projections...")
    if n_jobs > 1:
        try:
            with Pool(n_jobs) as pool:
                for i, (idx, result) in enumerate(pool.imap(process_projection, tasks)):
                    prim[:,:,idx] = result
                    if i % 50 == 0:
                        print(f"{i}/{proj.shape[2]}")
        except Exception as e:
            print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            # Fall back to sequential processing
            for i, proj_slice in enumerate(tasks):
                idx, result = process_projection(proj_slice)
                prim[:,:,idx] = result
                if i % 50 == 0:
                    print(f"{i}/{proj.shape[2]}")
    else:
        # Sequential processing
        for i, task in enumerate(tasks):
            idx, result = process_projection(task)
            prim[:,:,idx] = result
            if i % 50 == 0:
                print(f"{i}/{proj.shape[2]}")
    
    return prim


def sc_group_mask_vec(thickness, ngroup, nbounds):
    """
    Vectorized version of group mask generation
    """
    # Initialize masks
    nmask = np.zeros((thickness.shape[0], thickness.shape[1], ngroup))
    
    # Create masks for all groups except the last one (vectorized)
    for i in range(ngroup-1):
        nmask[:,:,i] = (thickness > nbounds[i]) & (thickness < nbounds[i+1])
    
    # Last group
    nmask[:,:,ngroup-1] = thickness > nbounds[ngroup-1]
    
    return nmask


def sc_asg_kernel(sc_calib, geo, dus, dvs):
    """
    Anti-scatter grid response function
    
    Parameters:
    -----------
    sc_calib : dict
        Scatter calibration structure
    geo : dict
        Geometry structure
    dus : ndarray
        Downsampled u vector
    dvs : ndarray
        Downsampled v vector
    
    Returns:
    --------
    kernel : ndarray
        Anti-scatter grid
    """
    # Transmission modelling (vectorized)
    k = -0.15
    b = 1
    t_ratio = k * np.abs(dvs / 10) + b
    
    # Kernel: [nv, nu] (broadcasting)
    kernel = t_ratio[:, np.newaxis] * np.ones(len(dus))
    
    # Get efficiency from calibration
    efficiency = float(sc_calib['CalibrationResults']['ObjectScatterModels']['ObjectScatterModel'][0]
                       ['GridEfficiency']['LamellaTransmission'])
    
    # Apply minimum efficiency (vectorized)
    kernel = np.maximum(kernel, efficiency)
    
    return kernel


def sc_thickness_estimator(blk, page, sc_calib, step_du, step_dv):
    """
    Estimate Water-Equivalent Thickness (2D)
    
    Parameters:
    -----------
    blk : ndarray
        Total intensity, I_0
    page : ndarray
        Primary intensity, I_p
    sc_calib : dict
        Scatter calibration structure
    step_du : float
        Step size in u direction
    step_dv : float
        Step size in v direction
    
    Returns:
    --------
    thickness : ndarray
        Estimated object thickness
    """
    # mu H2O = 0.02 /mm
    mu_h2o = float(sc_calib['CalibrationResults']['Globals']['muH2O'])
    
    # Calculate thickness (unit: mm) (vectorized)
    eps = np.finfo(float).eps
    tmp = np.maximum(blk / np.maximum(page, eps), 0.0001)
    thickness = np.log(tmp) / mu_h2o
    
    # Fill holes by interpolation
    thickness[thickness < 0] = np.nan
    thickness = inpaint_nans(thickness)
    
    # Smooth the estimated thickness
    thickness = sc_smooth_thickness(thickness, sc_calib, step_du, step_dv)
    
    return thickness


def sc_smooth_thickness(thickness, sc_calib, step_du, step_dv):
    """
    Gaussian Filter to Smooth Estimated Thickness
    
    Parameters:
    -----------
    thickness : ndarray
        Estimated object thickness
    sc_calib : dict
        Scatter calibration structure
    step_du : float
        Step size in u direction
    step_dv : float
        Step size in v direction
    
    Returns:
    --------
    thickness : ndarray
        Smoothed thickness
    """
    # Get sigma values (unit: mm)
    sigma_u = float(sc_calib['CalibrationResults']['Globals']['AsymPertSigmaMMu'])
    sigma_v = float(sc_calib['CalibrationResults']['Globals']['AsymPertSigmaMMv'])
    
    # Convert to pixel units
    sigma_u_px = sigma_u / step_du
    sigma_v_px = sigma_v / step_dv
    
    # Gaussian filtering (vectorized)
    return ndimage.gaussian_filter(thickness, [sigma_v_px, sigma_u_px])


def sc_form_func(sc_calib, dugd, dvgd):
    """
    Thickness-based Multiple Group Form Function kernels
    
    Parameters:
    -----------
    sc_calib : dict
        Scatter calibration structure
    dugd : ndarray
        Downsampled u grid
    dvgd : ndarray
        Downsampled v grid
    
    Returns:
    --------
    gform : ndarray
        Form functions
    """
    # Get number of groups
    obj_scatter_models = sc_calib['CalibrationResults']['ObjectScatterModels']['ObjectScatterModel']
    ngroup = len(obj_scatter_models)
    
    # Unit conversion: mm -> cm
    unit_cvt = 1/10
    dugd_cm = dugd * unit_cvt
    dvgd_cm = dvgd * unit_cvt
    
    # Calculate squared grid (once)
    grid2 = dugd_cm**2 + dvgd_cm**2
    
    # Initialize form functions
    gform = np.zeros((dugd.shape[0], dugd.shape[1], ngroup))
    
    # Calculate form functions for each group
    for i in range(ngroup):
        fit_params = obj_scatter_models[i]['ObjectScatterFit']
        
        # Get parameters (unit: cm^(-1))
        sigma1 = float(fit_params['sigma1'])
        sigma2 = float(fit_params['sigma2'])
        
        # Unitless parameter
        B = float(fit_params['B'])
        
        # Form Function (vectorized)
        gform[:,:,i] = np.exp(-0.5 * grid2 / (sigma1**2)) + B * np.exp(-0.5 * grid2 / (sigma2**2))
    
    return gform


def sc_edge_response(thickness):
    """
    Calculate edge response function
    
    Parameters:
    -----------
    thickness : ndarray
        Estimated thickness
    
    Returns:
    --------
    edgewt : ndarray
        Edge weight function
    """
    # Binarize thickness (empirical threshold)
    edgewt = (thickness > 50).astype(float)
    tmpmask = edgewt.copy()
    
    # Create average filter (once)
    kernel_size = 25
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Apply filter multiple times
    for _ in range(5):
        edgewt = ndimage.convolve(edgewt, kernel, mode='nearest')
    
    # Calculate final weight
    tmp = tmpmask * edgewt
    tmp[tmp == 0] = np.nan
    
    # Rescale to [0.6, 1] range
    tmp_min = np.nanmin(tmp)
    tmp_max = np.nanmax(tmp)
    
    if tmp_max > tmp_min:
        tmp = 0.6 + 0.4 * (tmp - tmp_min) / (tmp_max - tmp_min)
    
    tmp[np.isnan(tmp)] = 0
    
    return tmp


def sc_amplitude_factor(blk, page, edgewt, sc_calib):
    """
    Calculate amplitude factor with Edge Response Function
    
    Parameters:
    -----------
    blk : ndarray
        Blank scan
    page : ndarray
        Primary intensity
    edgewt : ndarray
        Edge weight function
    sc_calib : dict
        Scatter calibration structure
    
    Returns:
    --------
    cfactor : ndarray
        Amplitude factors
    """
    # Get number of groups
    obj_scatter_models = sc_calib['CalibrationResults']['ObjectScatterModels']['ObjectScatterModel']
    ngroup = len(obj_scatter_models)
    
    # Calculate ratio term (vectorized)
    eps = np.finfo(float).eps
    term = (page + eps) / (blk + eps)
    logterm = -np.log(term)
    logterm[logterm < 0] = np.nan
    logterm = inpaint_nans(logterm)
    
    # Initialize amplitude factors
    cfactor = np.zeros((page.shape[0], page.shape[1], ngroup))
    
    # Calculate amplitude factors for each group
    for i in range(ngroup):
        fit_params = obj_scatter_models[i]['ObjectScatterFit']
        
        # Get parameters
        A = float(fit_params['A']) / 10  # unit: mm -> cm
        alpha = float(fit_params['alpha'])  # unitless
        beta = float(fit_params['beta'])  # unitless
        
        # Calculate amplitude factor with edge response function (vectorized)
        cfactor[:,:,i] = A * edgewt * (term)**(alpha) * (logterm)**(beta)
    
    return cfactor


def inpaint_nans(array, method=2):
    """
    Fill in NaN values in an array
    
    Parameters:
    -----------
    array : ndarray
        Input array with NaNs
    method : int
        Method to use for inpainting (0: nearest, 1: linear, 2: cubic)
        
    Returns:
    --------
    filled : ndarray
        Array with NaNs filled
    """
    # Create a mask of NaN values
    mask = np.isnan(array)
    
    # If no NaNs, return the original array
    if not np.any(mask):
        return array
    
    # Get coordinates of non-NaN values (only calculate once)
    y, x = np.indices(array.shape)
    xgood = x[~mask]
    ygood = y[~mask]
    zgood = array[~mask]
    
    # Get coordinates of NaN values
    xbad = x[mask]
    ybad = y[mask]
    
    # Interpolate
    if method == 0:
        interp_method = 'nearest'
    elif method == 1:
        interp_method = 'linear'
    else:
        interp_method = 'cubic'
    
    filled = array.copy()
    filled[mask] = griddata((xgood, ygood), zgood, (xbad, ybad), method=interp_method, fill_value=0)
    
    # If there are still NaN values (can happen with cubic), fill with nearest
    if np.any(np.isnan(filled)):
        mask2 = np.isnan(filled)
        xbad2 = x[mask2]
        ybad2 = y[mask2]
        filled[mask2] = griddata((xgood, ygood), zgood, (xbad2, ybad2), method='nearest', fill_value=0)
    
    return filled




def read_xim(path):
    """
    Read XIM file format
    
    Args:
        path: Path to XIM file
        
    Returns:
        Dictionary containing image data and properties
    """
    try:
        xim_obj = XIM(path)
        return {
            'image': np.array(xim_obj),
            'properties': xim_obj.properties
        }
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def load_xim_file(file_name):
    """
    Load a single XIM file
    
    Args:
        file_name: Path to XIM file
    
    Returns:
        tuple: Rotated image, angle, air normalization, filename
    """
    try:
        xim_data = read_xim(file_name)
        if xim_data is None:
            return None
            
        metadata = xim_data['properties']
        #KVSourceRtn = GantryRtn + 90 deg
        angle = metadata['KVSourceRtn']
        airnorm = metadata['KVNormChamber']

        image = np.array(xim_data['image'])
        image_flipped = np.flipud(image)
        #image_rotated = np.rot90(image, -1)
    
        return image_flipped, angle, airnorm, file_name
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def proj_loader(datafolder, thd=0):
    """
    Load projection data from XIM files using parallel processing
    
    Args:
        datafolder: Path to data folder
        thd: Threshold for angular sampling (default is 0)
        
    Returns:
        tuple: Projections, angles, air normalization values
    """
    # List all .xim files in the directory
    xim_files = glob(os.path.join(datafolder, "Acquisitions", "*", "*.xim"))

    if not xim_files:
        raise FileNotFoundError(f"No .xim files found in {datafolder}/Acquisitions/*/")

    # Use Pool to load files concurrently
    with Pool(processes=16) as pool:  # Adjust number of processes as needed
        result_list = list(tqdm(pool.imap(load_xim_file, xim_files), 
                               total=len(xim_files), 
                               desc="Loading projections"))

    # Filter out None results and sort by filename
    result_list = sorted([r for r in result_list if r is not None and r[0].shape == (768, 1024)], key=lambda x: x[3])

    # Separate projections and angles
    projections_list, angles, air_norm, filenames = zip(*result_list)
    print(filenames)
    print(angles)

    # Stack projections into a NumPy array
    proj = np.stack(projections_list, axis=0)
    
    # Convert lists to numpy arrays
    angles = np.array(angles)
    air_norm = np.array(air_norm)
    
    # Filter projections based on threshold if needed
    if thd > 0:
        valid_indices = [0]
        for i in range(1, len(angles)):
            if abs(angles[i] - angles[valid_indices[-1]]) > thd:
                valid_indices.append(i)
        
        proj = proj[valid_indices]
        angles = angles[valid_indices]
        air_norm = air_norm[valid_indices]
    
    # Reshape projections from (n, height, width) to (height, width, n)
    proj = np.transpose(proj, (1, 2, 0))
    
    print("Loaded projections shape:", proj.shape)
    
    return proj, angles, air_norm



class GeometryOwn(Geometry):
    def __init__(self, geo_dict):
        Geometry.__init__(self)
        for key, value in geo_dict.items():
            setattr(self, key, value)


def geometry_from_xml(datafolder: str, load_geo: bool = True) -> Tuple[GeometryOwn, ET.ElementTree]:
    """
    Load scan and reconstruction geometry from XML files.

    Parameters:
    -----------
    datafolder : str
        Folder containing XML files
    load_geo : bool, optional
        Whether to load geometry (default is True)

    Returns:
    --------
    tuple: Geometry object and scan XML tree
    """
    # Load Scan.xml
    scan_xml_path = os.path.join(datafolder, 'Scan.xml')
    if not os.path.exists(scan_xml_path):
        raise FileNotFoundError(f"Scan.xml not found in {datafolder}")
    
    scan_tree = ET.parse(scan_xml_path)
    scan_root = scan_tree.getroot()

    namespace = {'ns': scan_root.tag.split('}')[0].strip('{')} if '}' in scan_root.tag else {}
    
    # Helper function to find element with namespace
    def find_elem(xpath):
        if namespace:
            return scan_root.find(xpath, namespace)
        else:
            # For XML without namespace
            return scan_root.find(xpath.replace('ns:', ''))

    # Initialize geometry dictionary
    geo = {'mode': 'cone'}

    # Extract scan parameters
    geo['DSD'] = float(find_elem('.//ns:Acquisitions/ns:SID').text)
    geo['DSO'] = float(find_elem('.//ns:Acquisitions/ns:SAD').text)
    
    # Detector parameters
    geo['nDetector'] = np.array([
        int(find_elem('.//ns:Acquisitions/ns:ImagerSizeY').text),
        int(find_elem('.//ns:Acquisitions/ns:ImagerSizeX').text)
    ])
    
    geo['dDetector'] = np.array([
        float(find_elem('.//ns:Acquisitions/ns:ImagerResY').text),
        float(find_elem('.//ns:Acquisitions/ns:ImagerResX').text),
    ])
    
    geo['sDetector'] = geo['nDetector'] * geo['dDetector']
    
    # Detector offset
    offset = float(find_elem('.//ns:Acquisitions/ns:ImagerLat').text)
    geo['offDetector'] = np.array([0, offset])
    
    geo['offOrigin'] = np.zeros(3)
    geo['accuracy'] = 0.5

    # Try to load Reconstruction.xml for image parameters
    pattern = os.path.join(datafolder, "Reconstructions", "*", "Reconstruction.xml")
    recon_xml_path = glob(pattern)[0]
    if load_geo and os.path.exists(recon_xml_path):
        recon_tree = ET.parse(recon_xml_path)
        recon_root = recon_tree.getroot()
        ns = {'baden': 'http://baden.varian.com/cr.xsd'}

        # Image parameters from Reconstruction.xml
        geo['sVoxel'] = np.array([
            float(recon_root.find('.//baden:VOISizeZ', ns).text),
            float(recon_root.find('.//baden:VOISizeY', ns).text),
            float(recon_root.find('.//baden:VOISizeX', ns).text)
        ])

        matrix_size = int(recon_root.find('.//baden:MatrixSize', ns).text)
        slice_thickness = float(recon_root.find('.//baden:SliceThickness', ns).text)
        slice_no = round(geo['sVoxel'][2] / slice_thickness)

        geo['nVoxel'] = np.array([slice_no, matrix_size, matrix_size])
        geo['dVoxel'] = geo['sVoxel'] / geo['nVoxel']
        print("Sucessfuly loaded Reconstruction.xml")
    else:
        # Estimate acceptable image size
        print('Estimating acceptable image size...')
        geo['dVoxel'] = np.array([
            geo['dDetector'][0], 
            geo['dDetector'][0], 
            geo['dDetector'][1]
        ]) * geo['DSO'] / geo['DSD']

        geo['nVoxel'] = np.ceil([
            geo['nDetector'][0] + abs(geo['offDetector'][0]) / geo['dDetector'][0],
            geo['nDetector'][0] + abs(geo['offDetector'][0]) / geo['dDetector'][0],
            geo['nDetector'][1]
        ]).astype(int)

        geo['sVoxel'] = geo['nVoxel'] * geo['dVoxel']

    return GeometryOwn(geo), scan_tree



def load_scatter_calibration(datafolder):
    """Load scatter correction calibration parameters from XML"""
    # Find the calibration XML file
    pattern = os.path.join(datafolder, 'Calibrations', 'SC-*', 'Factory', 'Calibration.xml')
    xml_files = glob(pattern)
    
    if not xml_files:
        print(f"Warning: No calibration file found in {pattern}")
        return None
    
    src_filename = xml_files[0]
    
    # Read XML file and filter out comment lines
    with open(src_filename, 'r') as src_file:
        xml_content = ''.join(line for line in src_file if '--' not in line)
    
    # Parse XML to dictionary
    try:
        return xmltodict.parse(xml_content)['Calibration']
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None


def detector_point_scatter_correction(proj, geo, sc_calib=None):
    proj_corrected = proj.astype(np.float64, copy=False)

    # Default scatter parameters
    a0, a1, a2, a3, a4, a5 = 1.0, 0.0003097, 0.546567, 0.311273, 0.002472, -12.6607

    if sc_calib:
        try:
            globals_params = sc_calib['CalibrationResults']['Globals']['DetectorScatterModel']
            a1, a2, a3, a4, a5 = [float(globals_params[f'PScFit{i}']) for i in range(5)]
        except (KeyError, TypeError):
            print("Warning: Using default scatter correction parameters")

    cover_spr = 0.04
    ds_rate = 8  # Downsampling factor

    us = (np.arange(-geo.nDetector[1]//2 + 0.5, geo.nDetector[1]//2) * geo.dDetector[1]) / 10
    vs = (np.arange(-geo.nDetector[0]//2 + 0.5, geo.nDetector[0]//2) * geo.dDetector[0]) / 10

    dus = us[::ds_rate]  # Downsampled grid
    dvs = vs[::ds_rate]

    grid = np.sqrt(dus[:, None]**2 + dvs[None, :]**2)
    hd = a0 * (a1 * np.exp(-a2 * grid) + a3 * np.exp(-a4 * (grid - a5)**3))
    hd = cover_spr / np.sum(hd) * hd  # Normalize kernel

    for i in range(proj.shape[2]):
        downsampled = zoom(proj[:, :, i], 1/ds_rate, order=1)  # Fast downsampling
        scatter = fftconvolve(downsampled, hd, mode='same')  # FFT-based convolution
        scatter_full = zoom(scatter, ds_rate, order=3)  # Fast upsampling

        proj_corrected[:, :, i] -= scatter_full  # Apply correction

    proj_corrected[proj_corrected < 0] = np.finfo(float).eps  # Avoid negatives
    return proj_corrected




def blk_loader(datafolder: str, allwaysOneDim = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load blank (air) projections for Varian CBCT scanner
    
    Args:
        datafolder: Path to data folder
    
    Returns:
        tuple: Blank projections, blank air normalization
    """
    # Load Scan.xml
    scan_xml_path = os.path.join(datafolder, 'Scan.xml')
    if not os.path.exists(scan_xml_path):
        raise FileNotFoundError(f"Scan.xml not found in {datafolder}")
    
    scan_tree = ET.parse(scan_xml_path)
    scan_root = scan_tree.getroot()

    namespace = {'ns': scan_root.tag.split('}')[0].strip('{')} if '}' in scan_root.tag else {}
    
    # Helper function to find element with namespace
    def find_elem(xpath):
        if namespace:
            return scan_root.find(xpath, namespace)
        else:
            return scan_root.find(xpath.replace('ns:', ''))
    
    # Determine rotation direction
    start_angle = float(find_elem('.//ns:Acquisitions/ns:StartAngle').text)
    stop_angle = float(find_elem('.//ns:Acquisitions/ns:StopAngle').text)
    rtn_direction = 'CC' if stop_angle - start_angle > 0 else 'CW'
    
    # Determine TrueBeam version
    version_str = scan_root.get('Version', '')
    if '2.7' in version_str:
        version = 2.7
    elif '2.0' in version_str:
        version = 2.0
    else:
        raise ValueError('Unsupported TrueBeam version')
    

    #HARD CODE VERSION 2.0
    if allwaysOneDim:
        version = 2.0
    
    # Check Bowtie filter
    bowtie_text = find_elem('.//ns:Acquisitions/ns:Bowtie').text
    tag_bowtie = bowtie_text.find('None') != -1
    
    # Find blank projection files
    blk_files = []
    
    if not tag_bowtie:
        if version == 2.0:
            pattern = os.path.join(datafolder, 'Calibrations', 'AIR-*', '**', 'FilterBowtie.xim')
            blk_files = glob(pattern, recursive=True)
        elif version == 2.7:
            pattern = os.path.join(datafolder, 'Calibrations', 'AIR-*', '**', f'FilterBowtie_{rtn_direction}*.xim')
            blk_files = glob(pattern, recursive=True)
    else:
        pattern = os.path.join(datafolder, 'Calibrations', 'AIR-*', '**', 'Filter.xim')
        blk_files = glob(pattern, recursive=True)
    
    # Process blank projections based on version
    if version == 2.0:
        print("Using version 2.0")
        # Single blank projection for version 2.0
        if not blk_files:
            raise FileNotFoundError("No blank projection files found")
        
        filename = blk_files[0]
        tmp = read_xim(filename)
        tmp_props = tmp['properties']
        
        #blk = np.rot90(np.array(tmp['image']), -1)[np.newaxis, :, :]
        #blk = np.array(tmp['image'])
        blk = np.flipud(np.array(tmp['image']))
        blk_air_norm = np.array([tmp_props['KVNormChamber']])
        
        return blk, None, blk_air_norm
    
    elif version == 2.7:
        # Multiple blank projections for version 2.7
        blk = []
        blk_air_norm = []
        sec = []
        
        for filename in blk_files:
            tmp = read_xim(filename)
            tmp_props = tmp['properties']
            #blk.append(np.rot90(np.array(tmp['image']), -1))
            #blk.append(tmp['image'])
            blk.append(np.flipud(tmp['image']))
            
            blk_air_norm.append(tmp_props['KVNormChamber'])
            sec.append(tmp_props['GantryRtn'])
        
        # Convert lists to numpy arrays
        blk = np.array(blk)
        blk_air_norm = np.array(blk_air_norm)
        sec = np.array(sec)
        
        # Reshape blk from (n, height, width) to (height, width, n)
        blk = np.transpose(blk, (1, 2, 0))
        
        return blk, sec, blk_air_norm
    
    raise ValueError("Unsupported TrueBeam version")


def log_normal(proj: np.ndarray, angles: np.ndarray, air_norm: np.ndarray, 
               blk: np.ndarray, sec: np.ndarray, blk_air_norm: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic normalization to projection data
    
    Args:
        proj: Projection data
        angles: Projection angles
        air_norm: Air normalization values
        blk: Blank projection data
        sec: Seconds data
        blk_air_norm: Blank air normalization values
    
    Returns:
        Log-normalized projection data
    """
    # Create a copy of the projections to avoid modifying the original
    proj_lg = np.zeros_like(proj, dtype=np.float32)
    eps = np.finfo(float).eps
    
    # Version 2.0 - Single blank projection
    if sec.all() == None:
        for i in range(len(angles)):
            # Correction factor
            cf = air_norm[i] / blk_air_norm[0]
            # Log normalization
            proj_lg[:, :, i] = np.log(cf * blk / (proj[:, :, i] + eps) + eps)
    
    # Version 2.7 - Multiple blank projections
    else:
        # Adjust angles for interpolation
        interp_angles = angles - 90
        
        # Flip for interpolation if necessary
        if sec.size > 1 and sec[1] - sec[0] < 0:
            sec = np.flip(sec)
            blk_air_norm = np.flip(blk_air_norm)
            blk = np.flip(blk, axis=2)
        
        # Interpolate blank projections for each angle
        for i in range(len(interp_angles)):
            # Find indices and weights for interpolation
            idx, weights = interp_weight(interp_angles[i], sec)
            
            # Interpolate blank projection
            if isinstance(idx, np.ndarray):
                idx = idx.item()  # Convert to scalar if it's an array
                
            # Ensure idx is within valid range
            idx = max(0, min(idx, sec.size - 2))
            
            # Interpolate blank projection
            interp_blk = weights[0] * blk[:, :, idx] + weights[1] * blk[:, :, idx + 1]
            
            # Correction factor
            cf = air_norm[i] / (0.5 * blk_air_norm[idx] + 0.5 * blk_air_norm[idx + 1])
            
            # Log normalization
            proj_lg[:, :, i] = np.log(cf * interp_blk / (proj[:, :, i] + eps) + eps)
    
    return proj_lg


def interp_weight(x: float, X: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Calculate interpolation weights for given value x in array X
    
    Args:
        x: Value to interpolate
        X: Array of reference values
    
    Returns:
        tuple: Index of lower bound and weights for interpolation
    """
    # Ensure x is scalar
    x = np.asarray(x).item() if isinstance(x, np.ndarray) else x
    
    # Find index where x falls between X[idx] and X[idx+1]
    idx = np.searchsorted(X, x, side='right') - 1
    
    # Clamp index to valid range
    idx = max(0, min(idx, len(X) - 2))
    
    # Calculate weights
    if x <= X[0]:
        weights = np.array([1.0, 0.0])
    elif x >= X[-1]:
        weights = np.array([0.0, 1.0])
    else:
        # Linear interpolation weights
        weights = np.array([
            (X[idx + 1] - x) / (X[idx + 1] - X[idx]),  # weight for X[idx]
            (x - X[idx]) / (X[idx + 1] - X[idx])       # weight for X[idx+1]
        ])
    
    return idx, weights


def ring_removal(proj, kernel_size=9):
    """
    Column-order based filtering for ring artifact removal in CT projections
    
    Parameters:
    -----------
    proj : ndarray
        The projection matrix (2D or 3D)
    kernel_size : int, optional
        Filter kernel size (default: 9)
        
    Returns:
    --------
    ndarray
        Filtered projection matrix

    """
    # Create a copy to avoid modifying the original
    proj_filtered = proj.copy()
    
    # Median filter implementation
    # For 3D projection matrix
    if proj.ndim == 3:
        for i in range(proj.shape[2]):
            # Create a horizontal 1D kernel (1 × kernel_size)
            proj_filtered[:, :, i] = median_filter(proj[:, :, i], footprint=np.ones((1, kernel_size)))
    else:
        # For 2D projection
        proj_filtered = median_filter(proj, footprint=np.ones((1, kernel_size)))

    return proj_filtered



def enforce_positive(proj: np.ndarray) -> np.ndarray:
    """
    Enforce positive values in projection data
    
    Args:
        proj: Projection data
    
    Returns:
        Projection data with negative values set to zero
    """
    return np.maximum(proj, 0)




def varian_data_loader(datafolder: str, tag_acdc: bool = True, tag_dps: bool = True, tag_sc: bool = False, tag_bh: bool = True, tag_ring: bool = True) -> Tuple[np.ndarray, GeometryOwn, np.ndarray]:
    """
    Loads Varian CBCT projection, geometry and angles data
    
    Args:
        datafolder: Path to data folder
        tag_acdc: Flag for acceleration/deceleration correction
        tag_dps: Flag for detector point scatter correction
        tag_sc: Flag for scatter correction
        tag_bh: Flag for beam hardening correction
        
    Returns:
        tuple: Log-normalized projections, geometry object, angles
    """

    # Load geometry
    print("Loading geometry")
    geo, scan_xml = geometry_from_xml(datafolder)

    #Load scanner scatter calibration
    if (tag_dps or tag_sc):
        ScCalib = load_scatter_calibration(datafolder)
        print(ScCalib)

    
    # Remove over-sampled projections due to acceleration and deceleration
    thd = 0
    if tag_acdc:
        # Extract from XML tree properly
        
        scan_root = scan_xml.getroot()
        ns = {'ns': scan_root.tag.split('}')[0].strip('{')} if '}' in scan_root.tag else {}
        velocity = float(scan_root.find('.//ns:Acquisitions/ns:Velocity', ns).text)
        frame_rate = float(scan_root.find('.//ns:Acquisitions/ns:FrameRate', ns).text)
        print("Vel/Framerate", velocity, frame_rate)
        angular_interval = velocity / frame_rate
        thd = angular_interval * 0.9

    # Load proj and angles
    print('Loading Proj: ')
    proj, angles, air_norm = proj_loader(datafolder, thd)
    print("Proj min/max: ", np.min(proj), np.max(proj))
    print("Proj mean: ", np.mean(proj))


    #Scatter correction
    if tag_dps:
        print('Detector point Scatter correction: ')
        proj = detector_point_scatter_correction(proj, geo, ScCalib)


    # Load blank scan
    print('Loading Blk: ')
    blk, sec, blk_air_norm = blk_loader(datafolder)
    print("Blank min/max: ", np.min(blk), np.max(blk))
    print("Blank mean: ", np.mean(blk))

    #Blank detector scatter correction
    if tag_dps:
        print('Blank Detector point scatter correction: ')
        blk = detector_point_scatter_correction(blk, geo, ScCalib)

    if tag_sc:
        print("Scatter correction")
        proj = scatter_correction(ScCalib, blk, blk_air_norm, proj, air_norm, geo)
        print("Scatter correction complete")
    
    proj = enforce_positive(proj)


    # Airnorm and Logarithmic Normalization
    proj_lg = log_normal(proj, angles, air_norm, blk, sec, blk_air_norm)
    print('Log Normalization is completed.')
    print("Proj min/max: ", np.min(proj_lg), np.max(proj_lg))
    
    # Remove anomalies
    proj_lg = enforce_positive(proj_lg)

    if tag_ring:
        #Ring removal
        print("Ring removal")
        proj_lg = ring_removal(proj_lg)

        # Remove anomalies
        proj_lg = enforce_positive(proj_lg)

    # double to single
    proj_lg = proj_lg.astype(np.float32)
    angles = np.deg2rad(angles)

    print('Data processing is complete! Ready for reconstruction: ')
    return proj_lg, geo, angles

