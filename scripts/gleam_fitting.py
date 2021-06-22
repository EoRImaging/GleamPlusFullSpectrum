from pyradiosky import SkyModel, utils
import numpy as np
from astropy.table import Table, setdiff, QTable
from astropy.utils.diff import report_diff_values
from astropy.io import fits
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import psutil
import erfa
import csv
import matplotlib.gridspec as gridspec
sm = SkyModel()

ruby_catalog = sm.from_fhd_catalog("/Users/Kiana1/uwradcos/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav",
                                  expand_extended = True)
gleam_catalog = sm.from_gleam_catalog("/Users/Kiana1/uwradcos/gleam.vot", spectral_type = "subband", with_error = True)
gleam_spectral_index = sm.from_gleam_catalog("/Users/Kiana1/uwradcos/gleam.vot", spectral_type = "spectral_index", with_error = True)


def log_linear_fit(freqs, fit_data, stokes_error, dec, detect_outlier = False):
    ## Compute combined error, fit logged data to linear polynomial, calculate chi2 residual
    # Calculate coord-based portion of error
    if (dec >= 18.5) or (dec <= -72):
        loc_error = fit_data * .03
    else:
        loc_error = fit_data * .02

    # Compute total error and weight for the polyfit
    total_error = np.sqrt(loc_error**2 + stokes_error**2)
    weight = np.log10(1 / total_error)

    # Take logs of data and freqs for polyfit
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

    #compute chi2 value
    variance = total_error[idx]**2
    residual = fit_data[idx] - fitted_data
    chi2 = sum((residual**2) / variance)
    chi2_residual = chi2 / (len(freqs[idx]) - 2)

    fitted_freqs = freqs[idx]
    fit_data_selected = fit_data[idx]

    original_parameters = float("NaN")
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
        if chi2_residual_ol < chi2_residual / 2:
            print(source, "GOOD:", chi2_residual, chi2_residual_ol)

            original_parameters = np.array([coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs, fit_data_selected], dtype=object)

            #reassign values with outlier removed version of fit
            chi2_residual = chi2_residual_ol
            coeffs = coeffs_ol
            fitted_data = fitted_data_ol
            all_freqs_fitted_data = all_freqs_fitted_data_ol
            fitted_freqs = np.delete(freqs[idx], idx_outlier)
            fit_data_selected = np.delete(fit_data[idx], idx_outlier)

    return(coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs, fit_data_selected, original_parameters)


## WORKING VERSION
# Initialize arrays and lists and stuff
source_dict = {}
nan_error = []
bad_chi2 = []
averages = []
fit_averages = []

# Separate all rows that contain nans
for source in np.arange(gleam_catalog.Ncomponents):
#for source in np.arange(1000):
    fit_data = gleam_catalog.stokes.value[0,:,source]
    dec = gleam_catalog.dec.value[source]
    freqs = gleam_catalog.freq_array.value
    stokes_error = gleam_catalog.stokes_error.value[0,:,source]

    averages.append(np.average(fit_data))

    # remove negative data by turning into nans before fitting, locate non-nan stokes values
    fit_data[fit_data < 0] = np.nan
    indices = np.argwhere(~np.isnan(fit_data)).flatten()

    #temporary statement to deal with the stokes nans
    if np.isnan(np.sum(stokes_error)):
        nan_error.append(source)
        continue

    out1 = log_linear_fit(freqs, fit_data, stokes_error, dec, detect_outlier = True)
    out = out1
    # if chi2_residual is >=1.93, fit again
    if out[1] >= 1.93:
        if len(fit_data[indices]) >= 8:
            half_freqs = freqs[indices[:int(len(indices) / 2)]]
            fit_data_half = fit_data[indices[:int(len(indices) / 2)]]
            error_half = stokes_error[indices[:int(len(indices) / 2)]]

            out2 = log_linear_fit(half_freqs, fit_data_half, error_half, dec)
            out = out2

            # if second fit isn't good try again
            if out[1] >= 1.93:
                # If 8+ freqs remain after halving, fit on bottom 1/4
                if len(half_freqs) >= 8:
                    qt_freqs = half_freqs[:int(len(half_freqs) / 2)]
                    fit_data_qt = fit_data_half[:int(len(half_freqs) / 2)]
                    error_qt = error_half[:int(len(half_freqs) / 2)]

                    out3 = log_linear_fit(qt_freqs, fit_data_qt, error_qt, dec)
                    out = out3

                # If there are <8 total non-nan frequencies, fit on bottom 4
                else:
                    bottom_freqs = freqs[indices[:4]]
                    fit_data_bottom = fit_data[indices[:4]]
                    error_bottom = stokes_error[indices[:4]]

                    out3 = log_linear_fit(bottom_freqs, fit_data_bottom, error_bottom, dec)
                    out = out3

        else:
            # run fit on bottom 4 non-nan-value frequencies if <8 total non-nan-value freqs
            bottom_freqs = freqs[indices[:4]]
            fit_data_bottom = fit_data[indices[:4]]
            error_bottom = stokes_error[indices[:4]]

            out2 = log_linear_fit(bottom_freqs, fit_data_bottom, error_bottom, dec)
            out = out2

    # if chi2_residual is still large after all iterations
    if out[1] >= 1.93:
        bad_chi2.append([source, out1[3], out2[3], out3[3], out1[1], out2[1], out3[1]])

    fit_averages.append(np.average(out[3]))

    #out has the vars: coeffs, chi2_residual, fitted_data, all_freqs_fitted_data, fitted_freqs, fit_data_selected, original_parameters
    # Create dict with final vals
    source_vars = {
        "ra": gleam_catalog.ra.value[source],
        "dec": dec,
        "coefficients": out[0],
        "chi2_residual": out[1],
        "all_freqs_fitted_data": out[3],
        "freqs": freqs,
        "freqs_used_for_fit": out[4],
        "data_used_for_fit": out[5],
        "pre_outlier_removal_output": out[6]
    }
    #print(source, source_vars['freqs_used_for_fit'])
    # source_dict is a dict of dicts
    source_dict[source] = source_vars

fit_output = pd.DataFrame(source_dict).T
fit_output.to_csv("/Users/Kiana1/uwradcos/GleamPlusFullSpectrum/misc/gleam_catalog_fits.csv")

def plotFits(gleam_catalog, source_dict, bad_chi2, savefig=False, Nsources=9, title="/Users/Kiana1/uwradcos/plots/bad_chi2_spectrums.png"):
    fig = plt.figure(figsize=(20,8*Nsources//3))
    #     print(2*(Nsources//3))
    nrows = int(np.ceil(Nsources/3))
    gs = gridspec.GridSpec(3*nrows, 3, height_ratios=np.tile([1,0.4,0.25],nrows))
    plt.subplots_adjust(hspace=.0)

    #     axs = axs.ravel()

    for i in range(Nsources):
        fit = plt.subplot(gs[(i//3)*9 + i%3])

        source_num = bad_chi2[i][0]
        freqs = source_dict[bad_chi2[i][0]]['freqs'] / 1000000
        first_fit = bad_chi2[i][1]
        second_fit = bad_chi2[i][2]
        final_fit = source_dict[bad_chi2[i][0]]['all_freqs_fitted_data']
        raw_data = gleam_catalog.stokes.value[0,:,source_num]
        residual = abs(raw_data / first_fit)

        #data plots
        fit.plot(freqs, first_fit, label = "first fit", color = "green")
        fit.plot(freqs, second_fit, label = "second fit", color = "orange")
        fit.plot(freqs, final_fit, label = "final fit", color = 'red')
        fit.scatter(freqs, raw_data, label = "Raw data")
        fit.set_title("Source "+ str(source_num) + ", RA " + "{:.2f}".format(source_dict[bad_chi2[i][0]]['ra']) + ", Dec " + "{:.2f}".format(source_dict[bad_chi2[i][0]]['dec']))


        #floating box
        textstr = '\n'.join((
        r'$chi1=%.2f$' % (bad_chi2[i][4]),
        r'$chi2=%.2f$' % (bad_chi2[i][5]),
        r'$chi3=%.2f$' % (bad_chi2[i][6])))

        props = dict(boxstyle = "round", facecolor = "wheat", alpha = .5)
        fit.text(0.97, 0.95, textstr, horizontalalignment='right', verticalalignment='top', transform=fit.transAxes, bbox = props)
        fit.set_ylim([np.nanmin(gleam_catalog.stokes.value[0,:,source_num])*.85, np.nanmax(gleam_catalog.stokes.value[0,:,source_num]) * 1.2])
        fit.axhline(linewidth=4, color="black")

        #fit.set_title(i)
        fit.set_xticks([])
        resid = plt.subplot(gs[(i//3)*9 + i%3 + 3])
        resid.scatter(freqs, residual, color = "black")


    fig.legend(axs,     # The line objects
           labels=["fit 1", "fit 2", "fit 3, final", "raw data", "residual"],   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Dataset"  # Title for the legend
           )
    fig.text(0.5, 0.1, 'Frequency (MHz)', ha='center', fontsize = 12)
    fig.text(0.08, 0.5, 'Flux (Jy)', va='center', rotation='vertical', fontsize = 12)
    fig.text(.5, .9, "Sample spectrums of bad chi2 fits", fontsize = 14)


# This is to plot sources that have had an outlier removed, before and after removal
def plotOutlierFits(gleam_catalog, source_dict, outlier_sources, savefig=False, Nsources=9, title="/Users/Kiana1/uwradcos/plots/compare_outlier_spectrums.png"):
    fig = plt.figure(figsize=(20,8*Nsources//3))
    #     print(2*(Nsources//3))
    nrows = int(np.ceil(Nsources/3))
    gs = gridspec.GridSpec(3*nrows, 3, height_ratios=np.tile([1,0.4,0.25],nrows))
    plt.subplots_adjust(hspace=.0)

    #     axs = axs.ravel()

    for i in range(Nsources):
        fit = plt.subplot(gs[(i//3)*9 + i%3])

        source_num = outlier_sources[i]

        # This is freqs that have no nans
        pre_outlier_freqs = source_dict[source_num]['pre_outlier_removal_output'][4] / 1000000
        post_outlier_freqs = source_dict[source_num]['freqs'] / 1000000

        raw_flux = source_dict[source_num]['data_used_for_fit']
        pre_outlier_flux = source_dict[source_num]['pre_outlier_removal_output'][2]
        post_outlier_flux = source_dict[source_num]['all_freqs_fitted_data']

        # identify the outlier
        outlier_freq = np.setdiff1d(np.union1d(pre_outlier_freqs, post_outlier_freqs), np.intersect1d(pre_outlier_freqs, post_outlier_freqs))
        outlier_flux = gleam_catalog.stokes.value[0,gleam_catalog.freq_array.value == outlier_freq * 1000000, source_num]

        raw_data = gleam_catalog.stokes.value[0,:,source_num]
        #residual = abs(raw_data / first_fit)

        #data plots
        fit.scatter(outlier_freq, outlier_flux, label = "outlier", color = "red")
        fit.plot(pre_outlier_freqs, pre_outlier_flux, label = "with outlier", color = "green")
        fit.plot(post_outlier_freqs, post_outlier_flux, label = "outlier removed", color = "orange")
        fit.scatter(gleam_catalog.freq_array.value, gleam_catalog.stokes.value[0,:,source_num], label = "raw data")
        fit.set_title("Source "+ str(source_num) + ", RA " + "{:.2f}".format(source_dict[source_num]['ra']) + ", Dec " + "{:.2f}".format(source_dict[source_num]['dec']))

        #floating box
        textstr = '\n'.join((
        r'$OutlierChi2=%.2f$' % (source_dict[source_num]['pre_outlier_removal_output'][1]),
        r'$noOutlierChi2=%.2f$' % (source_dict[source_num]['chi2_residual'])))

        props = dict(boxstyle = "round", facecolor = "wheat", alpha = .5)
        fit.text(0.97, 0.95, textstr, horizontalalignment='right', verticalalignment='top', transform=fit.transAxes, bbox = props)
        fit.set_ylim([np.nanmin(gleam_catalog.stokes.value[0,:,source_num])*.85, np.nanmax(gleam_catalog.stokes.value[0,:,source_num]) * 1.2])
        #fit.axhline(linewidth=4, color="black")

        #fit.set_title(i)
        fit.set_xticks([])
        #resid = plt.subplot(gs[(i//3)*9 + i%3 + 3])
        #resid.scatter(freqs, residual, color = "black")


    fig.legend(axs,     # The line objects
           labels=["outlier", "with outlier", "outlier removed", "raw data"],   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Dataset"  # Title for the legend
           )
    fig.text(0.5, 0.1, 'Frequency (MHz)', ha='center', fontsize = 12)
    fig.text(0.08, 0.5, 'Flux (Jy)', va='center', rotation='vertical', fontsize = 12)
    fig.text(.5, .9, "Before and after of outlier removal", fontsize = 14)

outlier_sources = []
#for i in np.arange(len(source_dict)):
for i in np.arange(19972):
    if type(source_dict[i]["pre_outlier_removal_output"]) != float:
        outlier_sources.append(i)
print(len(outlier_sources))

plotOutlierFits(gleam_catalog,source_dict,outlier_sources,savefig=False,Nsources=9)
