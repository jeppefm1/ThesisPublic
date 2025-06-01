import numpy as np
import matplotlib.pyplot as plt

# Define the data for each protocol
data = {
    "FDKHead": [
        [
            0.039,
            0.000853,
            0.017,
            0.019,
            0.020,
           0.028
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ],
    "MLEM50Head": [
        [
            0.039,
            0.008,
            0.018,
            0.019,
            0.020,
            0.027
            
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ],
    
    "FDKPelvis": [
        [
            0.035,
            0.003,
            0.016,
            0.018,
            0.019,
            0.025
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ],
    "MELM50Pelvis": [
        [
            0.035,
            0.008,
            0.017,
            0.018,
            0.019,
            0.025
            
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ],


    "FDKPelvisLarge": [
        [
            -0.014,
            -0.006,
            0.018,
            0.025
           
        ],
        [
            -1000.0,
            -196.65,
            -104.0,
            365.0
        ]
    ],
    "MELEM50PelvisLarge": [
            [
                0.033,
                0.008,
                0.017,
                0.018,
                0.019,
                0.024
            ],
            [
                1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
            ]
        ],
    
  
    "FDKChild": [
        [
            0.046,
            0.0025,
            0.018,
            0.019,
            0.021,
            0.031
            
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ],
    "MELEM50Child": [
        [
            0.045,
            0.007,
            0.018,
            0.019,
            0.020,
            0.030
        ],
        [
            1000.0,
            -1000.0,
            -196.65,
            -104.0,
            -47.0,
            365.0
        ]
    ]    }

# Function to fit a 2nd degree polynomial
def fit_polynomial(x, y):
    coefficients = np.polyfit(x, y, 2)  # Fit 2nd degree polynomial
    return coefficients

# Plotting the polynomial fits
for protocol, values in data.items():
    x = values[1]
    y = values[0]
    
    # Fit the polynomial
    coefficients = fit_polynomial(x, y)
    
    # Generate the fitted curve
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    
    # Plot the data and the fitted curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o', label=f'{protocol} data')
    plt.plot(x_fit, y_fit, '-', label=f'{protocol} fit')
    plt.xlabel('HU Target for material')
    plt.ylabel('Pixel value in reconstruction')
    plt.title(f'2nd Degree Polynomial Fit for {protocol}')
    plt.legend()
    plt.grid(True)
    plt.show()
