#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from photutils.aperture import  SkyCircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord

def coord(archivo, radius=0.4, frac_min=0.8):
    with fits.open(archivo, memmap=True) as hdu:
        header = hdu[1].header
        data = hdu[1].data  
    wcs = WCS(header).dropaxis(2)

    imagen_ref = np.nanmean(data, axis=0)
    n_y, n_x = imagen_ref.shape
    
    centros_validos = []
    for j in range(1, n_y-1, 3):
        for i in range(1, n_x-1, 3):
            vecinos = imagen_ref[j-1:j+2, i-1:i+2]
            if not np.isfinite(vecinos[1, 1]) or vecinos[1, 1] <= 0:
                continue

            position = SkyCoord.from_pixel(i, j, wcs) 
            aperture = SkyCircularAperture(position, r=radius * u.arcsec)
            pix_aperture = aperture.to_pixel(wcs)
            mask = pix_aperture.to_mask(method='center')
            cutout = mask.cutout(imagen_ref)
            
            if cutout is None:
                continue
            
            inside = mask.data > 0
            validos = np.isfinite(cutout) & (cutout > 0)

            frac_valida = np.sum(inside & validos) / np.sum(inside)

            if frac_valida >= frac_min:
                centros_validos.append(position)

    return centros_validos
