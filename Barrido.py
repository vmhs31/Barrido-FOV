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
    for j in range(n_y):
        for i in range(n_x):
            if imagen_ref[j, i] <= 0 or not np.isfinite(imagen_ref[j, i]):
                continue

            position = SkyCoord.from_pixel(i + 1, j + 1, wcs) 
            aperture = SkyCircularAperture(position, r=radius * u.arcsec)
            pix_aperture = aperture.to_pixel(wcs)
            mask = pix_aperture.to_mask(method='center')
            ap_mask = mask.to_image(imagen_ref.shape)
            inside = ap_mask > 0
            validos = np.isfinite(imagen_ref) & (imagen_ref > 0)

            if np.sum(inside) == 0:
                continue

            frac_valida = np.sum(inside & validos) / np.sum(inside)

            if frac_valida >= frac_min:
                centros_validos.append(position)

    return centros_validos
