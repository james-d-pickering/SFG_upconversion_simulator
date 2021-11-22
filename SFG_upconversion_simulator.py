#!/usr/bin/env python
# coding: utf-8

############################
# This program is designed to allow the effect of the shape/duration/width/Whatever of the 
# visible pulse in SFG spectroscopy to be seen on a given molecular resonance, and with a 
# given IR pulse. It doesn't account for realistic amplitudes and so shouldn't be used to 
# simulate actual spectra, but to get a feel for how different upconverters/pulses behave
# and can influence the spectra. It is mathematically correct as far as I can make it, but 
# I would still treat the results as semi-quantitative, as there are many other factors that
# contribute to spectral linewidth than only the upconverter. 

# Definitely mess around with all the functions and play with it. Let me know of any errors.

# Hope it's useful!

# - James D. Pickering, Aarhus University, 2021.
############################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import matplotlib.gridspec as gs
import warnings

warnings.filterwarnings("ignore") #JDP otherwise numpy complains if you plot the real part of a complex number

plt.close('all')

c = 3.0E8 #JDP speed of light m/s

##############################
#### Functions Start Here ####
##############################

def cm_to_hz(wavenumber):
    #JDP converting wavenumber to hertz
    freq = wavenumber * c * 100  
    return freq

def hz_to_cm(freq):
    #JDP converting hertz to wavenumber
    wavenumber = freq / (c*100)
    return wavenumber

def nm_to_hz(wavelength):
    #JDP converting metres to hertz
    freq = c/wavelength
    return freq

def gaussian(x, mean, FWHM, amp):
    #JDP generic gaussian
    c = FWHM/(2*np.sqrt(2*np.log(2)))
    gauss = amp*np.exp(-(x-mean)**2/(2*c**2))
    return gauss

def fourier_transform(signal, t_axis):
    #JDP fourier transforms the signal and shifts the frequencies so zero is in middle.
    #JDP function returns the power spectrum.
    fft = np.fft.fftshift(np.fft.fft(signal))
    
    #JDP getting freq axis in cm-1
    fft_freq = hz_to_cm(np.fft.fftshift(np.fft.fftfreq(np.size(t_axis), d=(t_axis[3]-t_axis[2]))))
    
    #JDP getting the power spectrum of the FFT
    fftpower = np.abs(fft)**2
    
    return fftpower, fft_freq
                               
def get_FFT_resolution_cm(freq_axis):
    #JDP get the spacing along the frequency axis as the inherent resolution in the FFT
    #JDP make sure it's smaller than what you want to measure
    
    freq_spacing = np.abs(freq_axis[1]-freq_axis[0])
    print("Inherent Resolution in the FFT", freq_spacing, "cm^-1")
    return freq_spacing

def get_FFT_peaks(signal, height, freq_axis, freq_spacing, resonant_width):
    #JDP simple peak finding to extract widths and stuff. Play with the height parameter a bit if
    #JDP needed. There is also way more functionality in the sp.signal package for other related stuff.
    peaks = sp.signal.find_peaks(signal, height=height)
    widths = sp.signal.peak_widths(signal, peaks[0][:])
    for x in range(np.size(peaks[0])):
        peak_freq = freq_axis[peaks[0][x]]
        peak_width = widths[0][x]*freq_spacing
        print("Peak found at ", np.round(peak_freq, decimals=3), "cm^-1.")
        print(" Upconverted Lorentzian width is", np.round(peak_width/2, decimals=3), "cm^-1")
        print(" True Resonant Lorentzian width is", np.round(resonant_width, decimals=3), "cm^-1")
    return

def generate_molecule_response( t_axis, t0, resonance_cm, lorentzian_width, onset_pulse_width, 
                               amp_resonance, amp_nonresonance, phase_resonance, phase_nonresonance):
    #JDP the wave on resonance - just a plane wave, but need the complex bits for FFT
    resonance_freq = cm_to_hz(resonance_cm)
    resonant_wave = np.exp(1j*resonance_freq*2*np.pi*t) 
    
    #JDP the lorentzian width. FFT of a lorenztian in frequency space is an exponential decay in time. 
    lorentzian_width_hz = cm_to_hz(lorentzian_width)
    decay_wave = np.exp(-lorentzian_width_hz*2*np.pi*t)

    #JDP the onset pulse - this is basically a step function as the gaussian FWHM is so narrow.
    onset_pulse = gaussian(t, t0, onset_pulse_width, 1)
    onset_pulse[t>t0] = 1
    
    #JDP non resonance - adding a non-resonant contribution if needed. With specific phase/envelope. Width follows the IR width.
    nonresonance_phase = np.exp(1j*phase_nonresonance)
    nonresonance_envelope = gaussian(t, t0, onset_pulse_width, amp_nonresonance)
    nonresonance = nonresonance_envelope*nonresonance_phase
    
    
    #JDP the total molecular response (which I called the FID for some reason, whatever...).
    #JDP scaling of the resonant part by the linewidth is left out as we don't really care. 
    FID = nonresonance-1j*onset_pulse*decay_wave*resonant_wave*amp_resonance*np.exp(1j*phase_resonance)
    return FID

def generate_IR_field( t_axis, t0IR, pulse_width, central_wavenumber, amp_IR ): 
    #JDP generating a simple IR field that is assumed to be a transform-limited Gaussian pulse with a given width.
    central_freq = cm_to_hz(central_wavenumber)
    envelope = gaussian(t_axis, t0IR, pulse_width, amp_IR)
    carrier = np.exp(1j*central_freq*2*np.pi*t_axis)
    E_IR = envelope*carrier
    return E_IR
    
def tophat(x, base_level, hat_level, hat_mid, hat_width):
    #JDP thing I stole off stackoverflow to make a tophat function
    return np.where((hat_mid-hat_width/2. < x) & (x < hat_mid+hat_width/2.), hat_level, base_level)
    
def generate_visible_field( t_axis, t0vis, vis_width_wavenumber, vis_central_wavelength, amp_vis ):
    #JDP generates an electric field for visible upconversion. 
    #JDP Various things in here that can be uncommented as needed.
    frequency_vis = nm_to_hz(vis_central_wavelength)
    width_vis = cm_to_hz(vis_width_wavenumber)
    
    ##################
    #JDP Gaussian pulse shape (as from SHBC). 
    #JDP Frequency width is defined by the temporal width of the Gaussian, assuming transform limited. 
    gauss_width_vis = 0.441/width_vis
    E_vis = amp_vis*np.exp(1j*frequency_vis*2*np.pi*t_axis)*gaussian(t_axis, t0vis, gauss_width_vis, 1)

    ##################
    
    ##################
    #JDP using a sinc function (4f)
   # sinc_vis = np.sin(width_vis*2*np.pi*(t_axis-t0vis))/(t_axis-t0vis)
    #E_vis = amp_vis * sinc_vis * np.exp(1j*frequency_vis*np.pi*2*(t_axis))
    ##################


    ##################
    #JDP true etalon pulse (takes ages to compute)
   # R = 0.955 #JDP reflectivity of etalon mirror
    #d = 0.01E-3 #JDP spacing between mirrors - this should give an etalon with about 15cm-1 resolution
    #c = 3.0E8
    #tau_in = 100E-15 #JDP width of input gaussian pulse

    #amp_etalon = 1
    #tau_RT = (2*d)/c
    #n_terms = 10000
    #series = 0
    
    #for n in range(n_terms):
    #   Rn  = R**n
    #   e1 = np.exp(-((t_axis-n*tau_RT)**2)/(tau_in**2))
    #   e2 = np.exp(1j*frequency_vis*2*np.pi*(t_axis-n*tau_RT))
    #   series = series + Rn*e1*e2
    #E_vis = amp_etalon*(1-R)*series
    ##################
    
    ##################
    #JDP fake etalon pulse which is faster to compute
   # decay_time = 1/(width_vis) #JDP roughly accurate..
    #onset_pulse = gaussian(t_axis, t0vis, 1E-15, 1)
    #onset_pulse[t>t0vis] = 1
    #E_vis = np.exp(1j*frequency_vis*2*np.pi*(t_axis-t0vis))*np.exp(-t_axis/decay_time)*onset_pulse
    ##################
    
    ##################
    #JDP monochromatic upconverter
   # E_vis = np.exp(1j*frequency_vis*2*np.pi*(t_axis))
    ##################
    return E_vis
    
def downconvert_freq_axis(freq_axis, vis_wavelength):
    #JDP shifts freq axis down by the upconverter frequency so it's in units of IR frequency
    vis_freq = hz_to_cm(nm_to_hz(vis_wavelength))
    freq_axis = freq_axis - vis_freq
    return freq_axis

def convolve_signals_pad(signal1, signal2, thresholdfactor1, thresholdfactor2, t_offset):
    #JDP do a numerical convolution cos the FFT one gives a load of noise and other BS
    #JDP but it is slower, so window out the bits we don't need to use and then zero pad to
    #JDP retain the FFT resolution (length of time axis dictates the resolution)
    
    #JDP finding the last element of each signal that isn't zero
    nonzero_1 = np.max(np.nonzero(signal1))
    nonzero_2 = np.max(np.nonzero(signal2))
    
    #JDP truncating the signal after this point by some factor. Did this just by trial and error and seeing
    #JDP when it gave no difference to doing a full convolution. There's maybe a smarter/faster way.
    threshold_1 = thresholdfactor1 * nonzero_1
    threshold_2 = thresholdfactor2 * nonzero_2
    
    #JDP window the signals up to the threshold
    signal1_window = signal1[0:threshold_1]
    signal2_window = signal2[0:threshold_2]
    
    #JDP convolve the signals directly using np.convolve. Mode='same' ensures that the length of the output
    #JDP is the same as the lengths of the inputs. 
    # convolution = np.convolve(signal1, signal2, mode='same')
    convolution = np.convolve(signal1_window, signal2_window, mode='same')
    
    #JDP convolution loses the t0 as the signal wasn't centered at zero (as it flips one of the signals)
    #JDP roll the array forward by the relevant time offset. This only works cos the long-time signal is zero,
    #JDP so this could be made more robust probably.
    convolution = np.roll(convolution, t_offset)

    #JDP zero pad the output back to the original length so we retain resolution in the FFT    
    pad_length = np.size(signal1)-np.size(signal1_window)
    padded_output = np.pad(convolution, (0, pad_length), 'constant', constant_values=(0))
    
    return padded_output, convolution




#############################
#### PROGRAM STARTS HERE ####
#############################


#JDP defining the time axis.
#JDP make t axis way longer than the decays so we don't get limited by the FFT resolution
t_spacing = 0.001 #in ps

#JDP doing the t axis like this seems to be more reliable than making it directly in ps for some reason
t_lower = -10 #JDP in ps
t_upper = 100 #JDP in ps
t = np.arange(t_lower, t_upper, t_spacing)*1E-12

#JDP define parameters to feed into the functions
IR_central_wavenumber = 1600 #JDP in cm-1
IR_temporal_width = 100E-15 #JDP in seconds
IR_t0 = 0 #JDP in seconds

vis_central_wavelength = 800E-9 #JDP in metres
vis_width_wavenumber = 10 #JDP in cm-1
vis_t0 = 0 #JDP in seconds

molecule_resonance_wavenumber = 1600 #JDP in cm-1
molecule_resonance_width_wavenumber = 15 #JDP in cm-1
resonant_amplitude = 1
nonresonant_amplitude = 1
resonant_phase = 0
nonresonant_phase = 0

#JDP generate an IR field to drive the FID
IREfield = generate_IR_field(t, IR_t0, IR_temporal_width, IR_central_wavenumber, 1)


#JDP generate a field for the upconverter
visEfield = generate_visible_field(t, vis_t0, vis_width_wavenumber, vis_central_wavelength, 1)


#JDP generating the response of the molecule 
molecule_response = generate_molecule_response(t, IR_t0, molecule_resonance_wavenumber, 
                                               molecule_resonance_width_wavenumber, IR_temporal_width, 
                                               resonant_amplitude, nonresonant_amplitude, resonant_phase, nonresonant_phase)

#JDP find the t0 offset of the IR pulse to shift the convolution, from whatever the lower limit of the time axis is.
t_offset = np.int(np.abs(t_lower)/t_spacing) #JDP this number is how many array elements to roll forward

#JDP the first order polarisation is the convolution of the molecular response and the driving IR pulse
polarisation_first_order, polarisation_nopad = convolve_signals_pad(molecule_response, IREfield, 6,6, t_offset)

#JDP the second order polarisation is the first order multiplied by the visible field 
polarisation_second_order = polarisation_first_order * visEfield

#JDP just say that the output E field to FFT is this polarisation. This is true up to a constant and a phase shift.
#JDP Could be probably more accurate but it's fine enough.
FID = polarisation_second_order

#JDP computing FFT and getting peak parameters
FID_fft_power, FID_freq = fourier_transform(FID, t)
fft_spacing = get_FFT_resolution_cm(FID_freq)
FID_freq_down = downconvert_freq_axis(FID_freq, vis_central_wavelength)
get_FFT_peaks(FID_fft_power, np.max(FID_fft_power)/10, FID_freq_down, fft_spacing, molecule_resonance_width_wavenumber)


#JDP now assume that we perfectly just converted the input FID, without any influence of the upconverter.
FID_perfect_power, FID_perfect_freq = fourier_transform(molecule_response, t)

#JDP plotting things
fig = plt.figure(figsize=(6,12))
grid = gs.GridSpec(2,1, hspace=0.2)
ax = fig.add_subplot(grid[0,0])
ax.set_title('Effect of the Upconversion')
ax.plot(FID_freq_down, FID_fft_power/np.max(FID_fft_power), label='With Upconversion', color='C0')
ax.plot(FID_perfect_freq, FID_perfect_power/np.max(FID_perfect_power), label='No Upconversion', color='C1')
ax.set_xlim(1400,1800)
ax.set_xlabel(r'Wavenumber [cm$^{-1}$]')
ax.set_ylabel(r'Spectral Power (normalised)')
ax.legend()

ax1 = fig.add_subplot(grid[1,0])
ax1.set_title('Normalised Electric Fields')
ax1.plot(t/1E-12, np.real(molecule_response)/np.max(molecule_response), label='Molecule Response', color='C0')
ax1.plot(t/1E-12, np.real(visEfield)/np.max(visEfield), label='Visible Field', color='C1', alpha=0.5)
ax1.plot(t/1E-12, np.real(IREfield)/np.max(IREfield), label='Infrared Field', color='C4', alpha=0.7)
ax1.plot(t/1E-12, np.real(polarisation_first_order)/np.max(polarisation_first_order), label='Molecule Response Convoluted with IR Field', color='C2', alpha=0.5)
#plt.plot(t/1E-12, np.real(polarisation_second_order)/np.max(polarisation_second_order), label='Upconverted Polarisation', color='C5', alpha=0.5)
ax1.set_xlabel(r'Time [ps]')
ax1.set_ylabel(r'Real Part of Electric Field [a.u.]')
ax1.set_xlim(t_lower,t_upper)
ax1.legend()

plt.show()



