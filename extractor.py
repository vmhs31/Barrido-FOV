#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from photutils.aperture import  SkyCircularAperture, SkyRectangularAperture, aperture_photometry
from astropy.coordinates import SkyCoord, Angle
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
from scipy.ndimage import center_of_mass

def center_of_mass_plane (data, wcs, mask=None, coords=True, center=None, rect=(10,10)):
        aux = np.copy(data)
        if mask is not None:
            aux[~(mask)] = 0.
        else:
            if center is not None:
                aux = aux * 0.
                x0 = int(center[0] - rect[0]/2)
                x1 = int(center[0] + rect[0]/2)
                y0 = int(center[1] - rect[1]/2)
                y1 = int(center[1] + rect[1]/2)
                aux[y0:y1, x0:x1] = np.copy(data[y0:y1, x0:x1])
                #aux[x0:x1, y0:y1] = np.copy(data[x0:x1, y0:y1])

        aux[np.isnan(aux)] = 0.
        cm = center_of_mass(aux)
        if not coords:
            return(cm)
        else:
            position = wcs.pixel_to_world(cm[1],cm[0])
            coord_cm = SkyCoord(position, frame='icrs', unit=(u.deg, u.deg))
            coord_cm = None if np.isnan(coord_cm.ra.deg) or np.isnan(coord_cm.dec.deg) else coord_cm
            return (coord_cm)

class EspectroExtractor:
    def __init__(self, archivos):
        self.archivos = archivos
    # num 0-5
    #0 es una apertura inicial de 0.13364089
    #1 es una apertura inicial de 0.26728177
    #2 es una apertura inicial de 0.4009227
    #3 es una apertura inicial de 0.53456354
    #4 es una apertura inicial de 0.6682045
    #5 es una apertura inicial de 0.8018454
    
    def ReadingApCorr(self, filecorr):
        import asdf
        with asdf.open(filecorr) as af:
            apcorr = af.tree['apcorr_table']['apcorr']
            errapcorr = af.tree['apcorr_table']['apcorr_err']
            apradius = af.tree['apcorr_table']['radius']
            apwv = af.tree['apcorr_table']['wavelength']
        
        return apcorr, errapcorr, apradius, apwv

    def extraer(self, RA, DEC, radius = None, filecorr = None, num=None):
        Flujos = []
        Longitudes = []

        if filecorr:
            (apcorr, errapcorr, apradius, apwv) =  self.ReadingApCorr(filecorr)
            for archivo in self.archivos:
                with fits.open(archivo) as hdu:
                    header = hdu[1].header
                    data = hdu[1].data
                wcs = WCS(header).dropaxis(2)
                position = SkyCoord(RA, DEC, unit='deg', frame='icrs')
                N_imagen = data.shape[0]
                Wl = header['CRVAL3'] + np.arange(N_imagen) * header['CDELT3']
                radio_corr = np.interp(Wl, apwv, apradius[num])
                corr_flux = np.interp(Wl, apwv, apcorr[num]) 
                pixscale = np.abs(header['CDELT1']) * (np.pi / 180)
                area_pix_sr = pixscale**2
                
                flujos = np.zeros(N_imagen)
                for i in range(N_imagen):
                    aperture = SkyCircularAperture(position, r=radio_corr[i] * u.arcsec)
                    pix_aperture = aperture.to_pixel(wcs)
                    imagen = data[i, :, :]
                    phot_table = aperture_photometry(imagen, pix_aperture)
                    flujos[i] = area_pix_sr*corr_flux[i] * phot_table['aperture_sum'][0]*1e9
                Longitudes.append(Wl)
                Flujos.append(flujos)
                
        else:
            for archivo in self.archivos:
                with fits.open(archivo) as hdu:
                    header = hdu[1].header
                    data = hdu[1].data
                wcs = WCS(header).dropaxis(2)
                position = SkyCoord(RA, DEC, unit='deg', frame='icrs')
                N_imagen = data.shape[0]
                Wl = header['CRVAL3'] + np.arange(N_imagen) * header['CDELT3']
                pixscale = np.abs(header['CDELT1']) * (np.pi / 180) 
                area_pix_sr = pixscale**2
                
                flujos = np.zeros(N_imagen)
                for i in range(N_imagen):
                    omega = np.pi*(radius*(np.pi/648000))**2
                    aperture = SkyCircularAperture(position, r=radius * u.arcsec)
                    pix_aperture = aperture.to_pixel(wcs)
                    imagen = data[i, :, :]
                    phot_table = aperture_photometry(imagen, pix_aperture)
                    flujos[i] = area_pix_sr*phot_table['aperture_sum'][0]*1e9
                Longitudes.append(Wl)
                Flujos.append(flujos)
                
        return Longitudes, Flujos

    @staticmethod
    def unir_grafica(longitudes1, longitudes2, flujos1, flujos2):
        longitudes1 = np.array(longitudes1)
        longitudes2 = np.array(longitudes2)
        flujos1 = np.array(flujos1)
        flujos2 = np.array(flujos2)

        p_pequena = np.abs(longitudes2 - longitudes1[-1]).argmin()
        p_grande = np.abs(longitudes1 - longitudes2[0]).argmin()

        long1_c = longitudes1[p_grande:]
        long2_c = longitudes2[:p_pequena]
        flujo1_c = flujos1[p_grande:]
        flujo2_c = flujos2[:p_pequena]

        flujo_interp = np.interp(long2_c, long1_c, flujo1_c)
        diferencia = np.mean(flujo_interp - flujo2_c)

        longitudes = np.concatenate([longitudes1, longitudes2[p_pequena:]])
        flujos = np.concatenate([flujos1, flujos2[p_pequena:] + diferencia])

        conjunto = list(zip(longitudes, flujos))
        conjunto_ordenado = sorted(set(conjunto), key=lambda x: x[0])
        longitudes_ordenadas = np.array([x[0] for x in conjunto_ordenado])
        flujos_ordenados = np.array([x[1] for x in conjunto_ordenado])

        return longitudes_ordenadas, flujos_ordenados

    def procesar(self, lista_longitudes, lista_flujos):
        resultado_long = [lista_longitudes[0]]
        resultado_flux = [lista_flujos[0]]

        for i in range(1, len(lista_longitudes)):
            long_merged, flujo_merged = self.unir_grafica(resultado_long[-1], lista_longitudes[i], resultado_flux[-1], lista_flujos[i])
            resultado_long.append(long_merged)
            resultado_flux.append(flujo_merged)

        return resultado_long[-1], resultado_flux[-1]
    
    def procesar_total(self, RA, DEC, apradius = None, filecorr = None, num=None):
        longitudes, flujos = self.extraer(RA, DEC, apradius, filecorr, num)
        long_final, flujo_final = self.procesar(longitudes, flujos)
        Longitudes_espectro = long_final[:np.abs(long_final - 27.7).argmin()]
        Flujo_espectro = flujo_final[:np.abs(long_final - 27.7).argmin()]
        
        return Longitudes_espectro, Flujo_espectro

def coord_small(archivo, RA, DEC, width, height, small_width = None):
    with fits.open(archivo) as cube:
        header = cube[1].header
    wcs = WCS(header).dropaxis(2)
    
    # Ángulo de posición de apertura utilizado
    APER_V3 = header['PA_V3'] + 7.6 #11
    angle = Angle(APER_V3, unit='deg')
    
    # Convertimos a SkyCoord y a pixeles
    centro = SkyCoord(RA*u.deg, DEC*u.deg)
    x_c, y_c = centro.to_pixel(wcs)
    
    # Definimos dimensiones en pixeles
    width_pix = (width * u.arcsec).to(u.deg)  # ancho en grados
    height_pix = (height * u.arcsec).to(u.deg)
    small_width_pix = wcs.proj_plane_pixel_scales()*u.deg

    # Cuántos pasos
    n_x = int(width_pix / small_width_pix[0])
    n_y = int(height_pix / small_width_pix[0])
    delta_pix  = 1
    New_center = []
    for i in range(n_x):
        for j in range(n_y):
            dx_pix = (-n_x/2 + 0.5 + i) * delta_pix
            dy_pix = (-n_y/2 + 0.5 + j) * delta_pix
            
            dx_pix_rot = dx_pix * np.cos(np.deg2rad(angle)) - dy_pix * np.sin(np.deg2rad(angle))
            dy_pix_rot = dx_pix * np.sin(np.deg2rad(angle)) + dy_pix * np.cos(np.deg2rad(angle))

            x_new = x_c + dx_pix_rot
            y_new = y_c + dy_pix_rot

            new_center = SkyCoord.from_pixel(x_new, y_new, wcs)
            New_center.append(new_center)

    return New_center

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

def paso_cte(wave, flux):
    Longitud = wave[:np.abs(wave - 27.5).argmin()]
    Flujo = flux[:np.abs(wave - 27.5).argmin()]
    LongitudesCte = np.arange(Longitud[0], Longitud[-1], np.abs(Longitud[-1]-Longitud[-2])) 
    FlujoCte = np.interp(LongitudesCte, Longitud, Flujo)     
    
    return LongitudesCte, FlujoCte

def quitar_lineas0(Longitud, Flujo, excepciones, ventana, n_std):
    if ventana % 2 == 0:
        ventana += 1

    Flujo0 = np.array(Flujo)
    mask_exc = np.ones_like(Flujo, dtype=bool)

    for idx in excepciones:
        start = np.abs(Longitud - idx[0]).argmin()
        end = np.abs(Longitud - idx[1]).argmin()
        mask_exc[start:end+1] = False
    Flujo0[~mask_exc] = 0

    #continuo = medfilt(Flujo0, ventana)
    continuo = median_filter(Flujo0, size=ventana, mode='nearest')
    wave, flux = paso_cte(Longitud, Flujo)
    _, flux_uniforme = paso_cte(Longitud, Flujo0)
    _, continuo_uniforme = paso_cte(Longitud, continuo)
    residuo = flux_uniforme - continuo_uniforme

    N = len(flux)
    half = ventana // 2
    peaks_global = []
    for i in range(half, N - half):
        local = residuo[i - half:i + half + 1]
        vecindad_clip = sigma_clip(local, sigma=3)
        sigma = np.std(vecindad_clip.data[~vecindad_clip.mask])
        if residuo[i] > n_std*sigma:
            peaks_global.append(i)

    mask = np.ones_like(flux_uniforme, dtype=bool)
    for p in peaks_global:
        i = p
        while i > 1 and flux_uniforme[i - 1] < flux_uniforme[i]:
            i -= 1
        j = p
        while j < len(flux_uniforme) - 2 and flux_uniforme[j + 1] < flux_uniforme[j]:
            j += 1
        mask[i:j+1] = False
        
    Flujo_filtrado = np.where(mask, flux, np.nan)
    
    if not np.all(mask):
        Flujo_interp = flux.copy()
        Flujo_interp[~mask] = np.interp(wave[~mask], wave[mask], Flujo_filtrado[mask])     
    else:
        Flujo_interp = flux

    return wave, flux, Flujo_interp

def snr_filtrado(Flujo, ventana):
    mediana = median_filter(np.nan_to_num(Flujo, nan=0.0), size=ventana, mode='nearest')
    SNR = np.zeros_like(Flujo, dtype=float)

    N = len(Flujo)
    half = ventana // 2
    for i in range(half, N - half):
        start = i - half
        end = i + half + 1

        ventana_flujo = Flujo[start:end]
        ventana_mediana = mediana[start:end]
        
        mascara = ~np.isnan(ventana_flujo)

        if np.sum(mascara) == 0:
            continue 

        senal = np.sum(np.abs(ventana_mediana[mascara]))
        ruido = np.sum(np.abs(ventana_flujo[mascara] - ventana_mediana[mascara]))
        SNR[i] = senal / np.maximum(ruido, 1e-8)
    
    return SNR
