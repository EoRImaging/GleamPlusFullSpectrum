from pyradiosky import SkyModel, utils
import numpy as np
from astropy.table import Table, setdiff
from astropy.utils.diff import report_diff_values
from astropy.io import fits
from operator import itemgetter
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import psutil
import erfa
import csv
sm = SkyModel()

gleam_catalog = sm.from_gleam_catalog("/Users/Kiana1/uwradcos/gleam.vot", spectral_type = "subband", with_error = True)
gleam_spectral_index = sm.from_gleam_catalog("/Users/Kiana1/uwradcos/gleam.vot", spectral_type = "spectral_index", with_error = True)

def log_linear_fit(freqs, fit_data, stokes_error, dec, detect_outlier = False):
    ## Compute combined error, fit logged data to linear polynomial, calculate chi2 residual

    # Calculate sky coordinate-based portion of error
    if (dec >= 18.5) or (dec <= -72):
        loc_error = fit_data * .03
    else:
        loc_error = fit_data * .02

    # Compute total error and weight for the polyfit
    total_error = np.sqrt(loc_error**2 + stokes_error**2)
    weight = np.log10(1 / total_error)

    # Convert data into log scale for polyfit
    fit_data_log = np.log10(fit_data)
    freqs_log = np.log10(freqs)
    all_freqs_log = np.log10(gleam_catalog.freq_array.value)

    # Subset to freqs with no nans in vals or errors and do polyfit on only those freqs
    idx = np.isfinite(freqs_log) & np.isfinite(fit_data_log) & np.isfinite(weight)

    # coeffs is a pair of numbers (b, m) corresponding to the equation y=b+mx
    coeffs = poly.polyfit(freqs_log[idx], fit_data_log[idx], w = weight[idx], deg=1)

    # Use coeffs to generate modeled vals at only freqs that were used to make coeffs
    fit_log = poly.polyval(freqs_log[idx], coeffs)
    fitted_data = 10**fit_log

    # use coefficients to generate modeled vals at all 20 freqs
    full_fit_log = poly.polyval(all_freqs_log, coeffs)
    all_freqs_fitted_data = 10**full_fit_log

    # generate modeled val at 50 MHz
    low_fit_log = poly.polyval(np.log10(50000000), coeffs)
    low_fit = 10**low_fit_log

    #compute chi2 value
    variance = total_error[idx]**2
    residual = fit_data[idx] - fitted_data
    chi2 = sum((residual**2) / variance)
    chi2_residual = chi2 / (len(freqs[idx]) - 2)

    fitted_freqs = freqs[idx]
    fit_data_selected = fit_data[idx]

    original_parameters = np.array([[float("NaN")]])
    # Outlier detection reruns fit without greatest outlier
    if detect_outlier == True:
        idx_outlier = np.argmax(abs(residual))

        # create datasets with outlier removed
        log_data_ol = np.delete(fit_data_log[idx], idx_outlier)
        log_freq_ol = np.delete(freqs_log[idx], idx_outlier)
        weight_ol = np.delete(weight[idx], idx_outlier)

        #fit without the outlier
        coeffs_ol = poly.polyfit(log_freq_ol, log_data_ol, w = weight_ol, deg=1)

        fit_log_ol = poly.polyval(log_freq_ol, coeffs_ol)
        fitted_data_ol = 10**fit_log_ol
        full_fit_log_ol = poly.polyval(all_freqs_log, coeffs_ol)
        all_freqs_fitted_data_ol = 10**full_fit_log_ol

        # compute chi2 using this new fit
        variance_ol = np.delete(total_error[idx], idx_outlier)**2
        residual_ol = np.delete(fit_data[idx], idx_outlier) - np.delete(fitted_data, idx_outlier)

        chi2_ol = sum((residual_ol**2) / variance_ol)
        chi2_residual_ol = chi2_ol / (len(np.delete(freqs[idx], idx_outlier)) - 2)


        # see if fit has improved
        if chi2_residual_ol < chi2_residual / 2.6:

            original_parameters = np.array([coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs, fit_data_selected], dtype=object)

            #reassign values with outlier removed version of fit
            chi2_residual = chi2_residual_ol
            coeffs = coeffs_ol
            fitted_data = fitted_data_ol
            all_freqs_fitted_data = all_freqs_fitted_data_ol
            fitted_freqs = np.delete(freqs[idx], idx_outlier)
            fit_data_selected = np.delete(fit_data[idx], idx_outlier)

    return(coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs, fit_data_selected, original_parameters, low_fit)



## COMPUTE FITS FOR SOURCES
# Initialize arrays and lists and stuff
source_dict = {}
bad_chi2 = []
fit_averages = []
problem_objs = []


# Separate all rows that contain nans
for source in np.arange(gleam_catalog.Ncomponents):
    fit_data = gleam_catalog.stokes.value[0,:,source]
    dec = gleam_catalog.dec.value[source]
    freqs = gleam_catalog.freq_array.value
    stokes_error = gleam_catalog.stokes_error.value[0,:,source]

    # Calculate variance for final output
    mean_adj_data = (fit_data - np.nanmean(fit_data)) / np.nanmean(fit_data)
    diff = np.diff(mean_adj_data)
    variance = np.nanvar(diff)

    #Initialize arrays for half and quarter fits
    out2 = np.array([[float("NaN")], [float("NaN")], [float("NaN")], [float("NaN")]])
    out3 = np.array([[float("NaN")], [float("NaN")], [float("NaN")], [float("NaN")]])

    # Find sources that have missing values in only one of error and vals
    source_probs = []
    for i in range(len(fit_data)):
        if np.isnan(fit_data[i]):
            if ~np.isnan(stokes_error[i]) and not source_probs:
                source_probs.append([fit_data, stokes_error])
        else:
            if np.isnan(stokes_error[i]) and not source_probs:
                source_probs.append([fit_data, stokes_error])

    # Only include in problems list if there WAS a problem, exclude source from rest of fitting
    if source_probs:
        problem_objs.append([source, gleam_catalog.ra.value[source], gleam_catalog.dec.value[source], source_probs])
        continue

    # Eliminate negative fluxes by turning into nans before fitting
    fit_data[fit_data < 0] = np.nan
    indices = np.argwhere(~np.isnan(fit_data)).flatten()

    # Skip sources with no values
    if np.all(np.isnan(fit_data)):
        continue


    # Perform full fit using all freqs available for source
    out1 = log_linear_fit(freqs, fit_data, stokes_error, dec, detect_outlier = True)
    out = out1

    # if chi2_residual is >=1.93 and brighter than 1Jy at 150MHz, fit again with fewer freqs
    if out[1] >= 1.93:
        if fit_data[9]>1:

            # Fit with bottom half of freqs
            if len(fit_data[indices]) >= 8:
                half_freqs = freqs[indices[:int(len(indices) / 2)]]
                fit_data_half = fit_data[indices[:int(len(indices) / 2)]]
                error_half = stokes_error[indices[:int(len(indices) / 2)]]

                # Fit with bottom half of freqs
                out2 = log_linear_fit(half_freqs, fit_data_half, error_half, dec)
                out = out2

                # if 2nd fit has poor chi2, fit with bottom 1/4 freqs
                if out[1] >= 1.93:
                    # If original freqs >=16, fit on bottom 1/4
                    if len(half_freqs) >= 8:
                        qt_freqs = half_freqs[:int(len(half_freqs) / 2)]
                        fit_data_qt = fit_data_half[:int(len(half_freqs) / 2)]
                        error_qt = error_half[:int(len(half_freqs) / 2)]

                        out3 = log_linear_fit(qt_freqs, fit_data_qt, error_qt, dec)
                        out = out3

                    # If there are <16 total non-nan frequencies, fit on bottom 4
                    else:
                        bottom_freqs = freqs[indices[:4]]
                        fit_data_bottom = fit_data[indices[:4]]
                        error_bottom = stokes_error[indices[:4]]

                        out3 = log_linear_fit(bottom_freqs, fit_data_bottom, error_bottom, dec)
                        out = out3

            else:
                # If bottom half of freqs is small, run fit on bottom 4 freqs, and do not attempt 3rd fit
                bottom_freqs = freqs[indices[:4]]
                fit_data_bottom = fit_data[indices[:4]]
                error_bottom = stokes_error[indices[:4]]

                # Fit with bottom half of freqs
                out2 = log_linear_fit(bottom_freqs, fit_data_bottom, error_bottom, dec)
                out = out2


    # if chi2_residual is still large after all iterations, take lowest one
    if out[1] >= 1.93:
        bad_chi2.append([source, out1[3], out2[3], out3[3], out1[1], out2[1], out3[1]])

        # select best of 3 fit options by chi2 val and use as final fit
        prev_rounds = {"out1": out1[1], "out2": out2[1], "out3": out3[1]}
        best_fit = min(prev_rounds, key=prev_rounds.get)
        out = eval(best_fit)


    fit_averages.append(np.average(out[3]))

    # These are the things contained in out1, out2, out3:
    # coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs,
    # fit_data_selected, original_parameters, low_fit

    # Create dict with final vals
    source_vars = {
        "ra": gleam_catalog.ra.value[source],
        "dec": dec,
        "coefficients": out[0],
        "chi2_residual": out[1],
        "prev_chi2_residuals": [out1[3], out2[3], out3[3], out1[1], out2[1], out3[1], out1[2], out2[2], out3[2],
                               out1[0]],
        "fitted_data": out[2],
        "all_freqs_fitted_data": out[3],
        "freqs": freqs,
        "freqs_used_for_fit": out[4],
        "data_used_for_fit": out[5],
        "pre_outlier_removal_output": out[6],
        "variance": variance,
        "50_mhz_extrapolation": [out[7], out1[7]]
    }
    # source_dict is a dict of dicts
    source_dict[source] = source_vars
