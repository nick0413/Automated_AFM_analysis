# First, we import the `TimeSeriesDict`
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# In order to generate a `FrequencySeries` we need to import the
# `~gwpy.timeseries.TimeSeries` and use
# :meth:`~gwpy.timeseries.TimeSeries.fetch_open_data` to download the strain
# records:

from gwpy.timeseries import TimeSeries
lho = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
llo = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)

# We can then call the :meth:`~gwpy.timeseries.TimeSeries.asd` method to
# calculated the amplitude spectral density for each
# `~gwpy.timeseries.TimeSeries`:
lhoasd = lho.asd(4, 2)
lloasd = llo.asd(4, 2)

# We can then :meth:`~FrequencySeries.plot` the spectra using the 'standard'
# colour scheme:

plot = lhoasd.plot(label='LIGO-Hanford', color='gwpy:ligo-hanford')
ax = plot.gca()
ax.plot(lloasd, label='LIGO-Livingston', color='gwpy:ligo-livingston')
ax.set_xlim(10, 2000)
ax.set_ylim(5e-24, 1e-21)
ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc='lower right', ncol=2)
plot.show()