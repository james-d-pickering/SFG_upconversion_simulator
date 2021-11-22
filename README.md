# SFG_upconversion_simulator
Simulates spectral lineshapes in SFG spectroscopy.

The SFG upconversion simulator is designed to allow you to generate the spectral line that a user-given molecular resonance will produce in an sum-frequency generation (SFG) spectroscopy measurement, with a given IR and VIS (upconversion) pulse. Specifically, the aim is to allow the effect of the delay between IR and VIS, and the effect of the VIS pulse shape/width/length on the output spectrum to be seen. The program is pretty thoroughly commented and is quite straightforward to use, it will output an illustration of your input electric fields, and also the "ideal" spectral line (with perfect upconversion), together with the "actual" spectral line that your given upconverter will create. 

It does not account for realistic amplitudes and so cannot simulate an actual spectrum quantitatively, but illustrates the effects on spectral linewidth well. I would treat the results as semi-quantitative, as there are many factors aside from the VIS pulse that contribute to spectral linewidth (it doesn't, for example, account for if you've focussed into your spectrograph badly). I think it is most useful as a teaching/pedagogical tool for people new to the field to mess around with.
