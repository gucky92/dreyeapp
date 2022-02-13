"""
Streamlit app for dreye
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import streamlit as st

import dreye


### --- caching functions

# TODO check if color (from labels) in matplotlib - then use that

appfolder = os.path.dirname(__file__)

# preload some data
@st.cache
def load_thorlabs():
    tl_file = os.path.join(appfolder, 'thorlabs.csv')
    return pd.read_csv(tl_file).set_index('wls')

thorlabs = load_thorlabs()

# globally cached parameters
@st.cache
def load_global_params():
    global_params = {
        'wls': np.arange(200, 801, 1), 
        'templates': [
            'govardovskii2000_template', 
            'gaussian_template', 
            'stavenga1993_template'
        ]
    }
    global_params['templates_formatted'] = {
        template.replace('_', ' ').capitalize(): template
        for template in global_params['templates']
    }
    return global_params


@st.cache
def load_filters(filter_choice, filters_file, peaks, wls_range):
    # Load wls, filters, labels
    if filter_choice is None:
        df = pd.read_csv(filters_file).set_index('wls')
        wls = df.index.to_numpy().astype(np.float64)
        filters = df.to_numpy().T.astype(np.float64)
        labels = df.columns.to_numpy()
    
    else:
        template_func = getattr(dreye, filter_choice)
        wls = np.arange(*wls_range, 1.0)
        filters = template_func(wls, np.array(peaks)[:, None])
        labels = np.arange(filters.shape[0])

    return wls, filters, labels


@st.cache
def load_sources(sources_file, wls, thorlabs_list):
    if thorlabs_list:
        df = thorlabs[thorlabs_list]
    else:
        df = pd.read_csv(sources_file).set_index('wls')

    sources_wls = df.index.to_numpy().astype(np.float64)
    sources = df.to_numpy().T.astype(np.float64)
    sources_labels = df.columns.to_numpy()
    sources = interp1d(
        sources_wls, sources, 
        axis=-1, bounds_error=False, 
        fill_value=0
    )(wls)
    # Shouldn't be called within
    if np.all(sources == 0, axis=-1).any():
        raise ValueError(
            "Light sources from CSV file have only zeros "
            "or are outside the wavelength range for the filters: "
            f"{sources}"
        )
    sources = sources / dreye.integral(sources, wls, axis=-1, keepdims=True)
    return sources, sources_labels


@st.cache
def load_filters_uncertainty(uncertainty_file, labels, wls):
     # Load wls, filters, labels
    df = pd.read_csv(uncertainty_file).set_index('wls')
    uwls = df.index.to_numpy().astype(np.float64)
    filters = df.loc[:, labels].to_numpy().T.astype(np.float64)
    
    if (wls.size != uwls.size) or not np.allclose(wls, uwls):
        filters = interp1d(
            uwls, filters, bounds_error=False, fill_value=1e-8, 
            axis=-1
        )(wls)
    return filters


@st.cache
def load_spectrum(spectrum_file, wls):
    if spectrum_file is None:
        return
    df = pd.read_csv(spectrum_file).set_index('wls')
    uwls = df.index.to_numpy().astype(np.float64)
    spectrum = df.to_numpy().T.astype(np.float64)
    assert len(spectrum) == 1, "More than one spectrum in file for background"
    spectrum = spectrum[0]
    
    if (wls.size != uwls.size) or not np.allclose(wls, uwls):
        spectrum = interp1d(
            uwls, spectrum, bounds_error=False, fill_value=0, 
            axis=-1
        )(wls)
    return spectrum
    
   
@st.cache
def load_receptor_estimator(
    filters, 
    wls, 
    uncertainty, 
    ws, labels, 
    K, baseline, 
    sources, max_ints, 
    sources_labels, 
    adapt_ints, 
    spectrum,
):
    est = dreye.ReceptorEstimator(
        filters=filters, 
        domain=wls, 
        filters_uncertainty=uncertainty, 
        w=ws, labels=labels, 
        K=(1 if K is None else K), 
        baseline=baseline, 
        sources=sources, lb=0, 
        ub=max_ints, 
        sources_labels=sources_labels
    )
    
    if adapt_ints is not None:
        est.register_system_adaptation(adapt_ints)
    if spectrum is not None:
        est.register_background_adaptation(spectrum)
    return est

### --- START

global_params = load_global_params()


# title of app
st.title("DrEye: Color Stimulus Design")

### --- the sidebar
estimator_loaded = False
filters_loaded = False
sources_loaded = False

st.sidebar.markdown("### Assign a set of opsin sensitivities")
# getting the filter
filter_choice = st.sidebar.selectbox(
    "How do you want to construct your opsin sensitivities?", 
    list(global_params['templates_formatted']) + ['from CSV file'],
)
filter_choice = global_params['templates_formatted'].get(filter_choice, None)
if filter_choice is None:
    filters_file = st.sidebar.file_uploader(
        "Upload a `.csv` file with header columns `wls` and opsin names.", 
        type='csv', 
        accept_multiple_files=False, 
    )    
    peaks = None
    wls_range = None
else:
    filters_file = None
    peaks = st.sidebar.multiselect(
        "Choose a set of opsin sensitivity peaks (in nm)", global_params['wls'],
    )
    wls_range = st.sidebar.select_slider(
        "Select relevant wavelength range (in nm)", 
        global_params['wls'][::5],  # steps of 5 nanometers 
        value=(300, 700), 
    )

# load filters
if filters_file is None and not peaks:
    wls = global_params['wls']
    filters_loaded = False
else:
    wls, filters, labels = load_filters(filter_choice, filters_file, peaks, wls_range)
    n_filters = len(filters)
    filters_loaded = True
    
if filters_loaded:
    st.sidebar.markdown("### Assign a set of light sources")
    
    sources_type = st.sidebar.selectbox(
        "How do you want to define your light sources?", 
        options=[
            "From Thorlabs",
            "From CSV file"
        ]
    )
    
    if sources_type == 'From Thorlabs':
        thorlabs_list = st.sidebar.multiselect(
            "Select ThorLabs hardware", 
            options=list(thorlabs.columns)
        )
        sources_file = None
    else:
        thorlabs_list = None
        sources_file = st.sidebar.file_uploader(
            "Upload a `.csv` file with header columns `wls` and light sources' names.", 
            type='csv', 
            accept_multiple_files=False, 
        )

    if sources_file is None and not thorlabs_list:
        sources_loaded = False
    else:
        sources, sources_labels = load_sources(sources_file, wls, thorlabs_list)
        n_sources = len(sources)        
        # max intensity
        st.sidebar.markdown("Max intensity of each light source (in units of flux).")
        cols = st.sidebar.columns(2)
        max_ints = []
        for idx, label in enumerate(sources_labels):
            col = cols[idx % 2]
            max_int = col.number_input(f"max for {label}", min_value=0.0, value=1.0, step=1.0)
            max_ints.append(max_int)
        max_ints = np.array(max_ints)
        sources_loaded = True
        
if filters_loaded and sources_loaded:
    st.sidebar.markdown("### Receptor model parameters")
    baseline = st.sidebar.number_input(
        "Baseline absolute capture value (aka dark count)", 
        min_value=0.0, 
        value=0.0,
        step=0.0001,
        format="%.4f"
    )
    
    adaptional_type = st.sidebar.selectbox(
        "How should the adaptation of photoreceptors be calculated?", 
        options=[
            'light source intensities', 
            'absolute captures', 
            'intensity spectrum (in units of flux)', 
        ]
    )
    if adaptional_type == 'light source intensities':
        cols = st.sidebar.columns(2)
        adapt_ints = []
        for idx, label in enumerate(sources_labels):
            col = cols[idx % 2]
            adapt_int = col.number_input(f"bg for {label}", min_value=0.0, value=1.0, step=1.0)
            adapt_ints.append(adapt_int)
            
        K = None
        spectrum = None
    elif adaptional_type == 'absolute captures':
        cols = st.sidebar.columns(2)
        K = []
        for idx, label in enumerate(labels):
            col = cols[idx % 2]
            k = col.number_input(f"adapt for {label}", min_value=0.0, value=1.0, step=1.0)
            K.append(k) 
        K = np.array(K)
        
        spectrum = None
        adapt_ints = None
    else:
        spectrum_file = st.sidebar.file_uploader(
            "Spectrum photoreceptors are adapted to. "
            "Upload a `.csv` file with header columns `wls` and `spectrum`. "
        )
        spectrum = load_spectrum(spectrum_file, wls)
        adapt_ints = None
        K = None
        
    
    # importance of each filter
    st.sidebar.markdown("Importance of each opsin.")
    cols = st.sidebar.columns(2)
    ws = []
    for idx, label in enumerate(labels):
        col = cols[idx % 2]
        w = col.number_input(f"weight for {label}", min_value=0.0, value=1.0, step=1.0)
        ws.append(w)
    ws = np.array(ws)
    
    uncertainty = st.sidebar.file_uploader(
        "Standard deviation of each sensitivity across wavelengths. "
        "Upload a `.csv` file with header columns `wls` and opsin names. "
        "If not given, then the standard deviation is equal to the provided sensitivities."
    )
    
    if uncertainty is not None:
        uncertainty = load_filters_uncertainty(uncertainty, labels, wls)
    
    est = load_receptor_estimator(
        filters, 
        wls, 
        uncertainty, 
        ws, labels, 
        K, baseline, 
        sources, max_ints, 
        sources_labels, 
        adapt_ints, 
        spectrum,
    )
    estimator_loaded = True
    

# load the various things

### --- the main window

if not estimator_loaded:
    st.error("REQUIRED: Setup the opsin sensitivity set and the light sources in the sidebar before proceeding")
    
    
with st.expander("Color Model Summary"):
    # pairwise plots 
    # gamut plot
    # capture heat matrix
    if estimator_loaded:
        filters_colors = sns.color_palette('Greys', len(filters)+2)[1:-1]
        
        fig, ax1 = plt.subplots()

        ax1 = est.filter_plot(colors=filters_colors, ax=ax1)
        ax1.set_xlabel('wavelength (nm)')
        ax1.set_ylabel('relative sensitivity (a.u.)')
        ax1.legend(loc=2)
    
        sources_colors = sns.color_palette('rainbow', len(sources))
        ax2 = plt.twinx(ax1)
        est.sources_plot(colors=sources_colors, ax=ax2)
        ax2.set_ylabel('relative intensity (1/nm)')
        ax2.legend(loc=1)
        
        st.pyplot(fig)
    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # if filters_loaded:
    #     for label, sensitivity in zip(labels, filters):
    #         filter_trace = go.Scatter(
    #             x=wls, 
    #             y=sensitivity,
    #             name=f"opsin {label}"
    #         )

    #         fig.add_trace(filter_trace)
            

    # if sources_loaded:
    #     for label, source in zip(sources_labels, sources):
    #         source_trace = go.Scatter(
    #             x=wls, 
    #             y=source, 
    #             name=f"{label}"
    #         )
            
    #         fig.add_trace(source_trace, secondary_y=True)
            
    # fig.update_layout(title_text='Spectral sensitivities and light sources')
    # fig.update_xaxes(title_text='wavelength (nm)', showgrid=True)
    # fig.update_yaxes(title_text='sensitivity', showgrid=True, secondary_y=False)
    # fig.update_yaxes(title_text='normalized intensity', showgrid=False, secondary_y=True)
            
    # st.plotly_chart(fig, use_container_width=True)
    
# registering values !!!
# TODO how to organize?

# set a bunch of points
# plot the points in gamut plot, etc. - visualization

    
with st.expander("Randomly sample points in gamut"):
    nsamples = st.number_input("Number of samples", min_value=0, value=1)
    
with st.expander("Analyze intensity values within the receptor model"):
    isets_file = st.file_uploader("Upload sets of intensities as a CSV File.")
    
with st.expander("Fit and analyze a set of capture values"):
    # randomly sample using a gaussian, uniform, qmc, etc.
    qsets_file = st.file_uploader("Upload sets of capture values")
    
with st.expander("Fit a hyperspectral image using the color model"):
    hyperspectral_file = st.file_uploader("Upload a hyperspectral image.")
    

# Layout right to left
# Goal is to get a set of intensities that 
# correspond to a desired stimulus
# choosing stimuli to fit or get
# sampling, 
# uploading hyperspectral image, 
# upload a set of capture values
# plot capture values from uploaded intensities