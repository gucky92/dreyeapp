"""
Streamlit app for dreye
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

import streamlit as st

import dreye

# ensure constrained layout
plt.rcParams['figure.constrained_layout.use'] = True


### --- caching functions

# TODO add unit conversion

appfolder = os.path.dirname(__file__)
AZIMUTH_DEFAULT = -45
ELEVATION_DEFAULT = 30
STANDARD_FLUX_SCALE = 10 ** 6 # micro
UNITS_OPTIONS = [
    'uE', 
    'E',
    'W/m^2', 
    'uW/cm^2'
]
UNITS_TEXT = "Units used in file (E=mol/s/m^2)"

# preload some data
@st.cache
def load_thorlabs():
    tl_file = os.path.join(appfolder, 'thorlabs.csv')
    return pd.read_csv(tl_file).set_index('wls')

thorlabs = load_thorlabs()

@st.cache
def register_and_fit(
    est, wls, sampling_method, gamut_correction, fit_method, 
    hull_samples, hull_seed, hull_l1, 
    spectra_sample_file, spectra_sample_units, 
    intensity_sample_file, intensity_sample_units, 
    l2_eps, n_layers
):
    csvs = {}
    data = {}
    B = None
    if sampling_method == 'sample within gamut':
        B = est.sample_in_hull(hull_samples, hull_seed, l1=(None if not hull_l1 else hull_l1))
    elif sampling_method == 'upload spectra':
        spectra_samples = load_spectra_samples(spectra_sample_file, wls, spectra_sample_units)
        B = est.relative_capture(spectra_samples)
    elif sampling_method == 'upload intensities':
        ints_samples = load_light_source_samples(intensity_sample_file, intensity_sample_units, est=est)
        B = est.system_relative_capture(ints_samples)
    else:
        raise RuntimeWarning(f"Sampling method not recognized: {sampling_method}")
    
    sample_index = pd.Index(range(len(B)), name='sample_number')
    
    # df.to_csv().encode('utf-8')
    csvs['target_captures'] = pd.DataFrame(B, columns=est.labels, index=sample_index).to_csv().encode('utf8')
    
    if gamut_correction == 'intensity scaling':
        data['Bbefore'] = B
        data['inhull_before'] = est.in_hull(B)
        B = est.gamut_l1_scaling(B)
        csvs['corrected_captures'] = pd.DataFrame(B, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        data['r2before'] = r2_score(data['Bbefore'], B, multioutput="raw_values")
    elif gamut_correction == 'chromatic scaling':
        data['Bbefore'] = B
        data['inhull_before'] = est.in_hull(B)
        B = est.gamut_l1_scaling(B)
        B = est.gamut_dist_scaling(B)
        csvs['corrected_captures'] = pd.DataFrame(B, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        data['r2before'] = r2_score(data['Bbefore'], B, multioutput="raw_values")
    elif gamut_correction == 'adaptive scaling':
        data['Bbefore'] = B
        data['inhull_before'] = est.in_hull(B)
        _, _, B = est.fit_adaptive(B, delta_norm1=1e-4, delta_radius=1e-3, adaptive_objective='max', scale_w=np.array([0.001, 10]))
        csvs['corrected_captures'] = pd.DataFrame(B, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        data['r2before'] = r2_score(data['Bbefore'], B, multioutput="raw_values")
        
    data['B'] = B
    data['inhull'] = est.in_hull(B)
    
    if fit_method in ['poisson', 'gaussian', 'excitation']:
        Xfit, Bfit = est.fit(B, model=fit_method)
        data['Xfit'] = Xfit
        data['Bfit'] = Bfit
        csvs['fitted_captures'] = pd.DataFrame(Bfit, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        csvs['fitted_intensities'] = pd.DataFrame(Xfit, columns=est.sources_labels, index=sample_index).to_csv().encode('utf8')
    elif fit_method == 'minimize variance':
        Xfit, Bfit, Bvar = est.minimize_variance(B, l2_eps=l2_eps)
        data['Bvar'] = Bvar
        data['Xfit'] = Xfit
        data['Bfit'] = Bfit
        csvs['fitted_captures'] = pd.DataFrame(Bfit, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        csvs['capture_variances'] = pd.DataFrame(Bvar, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        csvs['fitted_intensities'] = pd.DataFrame(Xfit, columns=est.sources_labels, index=sample_index).to_csv().encode('utf8')
    elif fit_method == 'decompose subframes':
        Xfit, Pfit, Bfit = est.fit_decomposition(B, n_layers=n_layers)
        layer_labels = [f"subframe_{i}" for i in range(1, n_layers+1)]
        data['Xfit'] = Xfit
        data['Bfit'] = Bfit
        data['Pfit'] = Pfit
        csvs['fitted_captures'] = pd.DataFrame(Bfit, columns=est.labels, index=sample_index).to_csv().encode('utf8')
        csvs['fitted_intensities'] = pd.DataFrame(Xfit, columns=est.sources_labels, index=pd.Index(layer_labels, name='layer_label')).to_csv().encode('utf8')
        csvs['subframe_intensities'] = pd.DataFrame(Pfit, columns=layer_labels, index=sample_index).to_csv().encode('utf8')
    else:
        raise RuntimeError(f"Fit method not recognize: {fit_method}")
    
    data['r2'] = r2_score(B, Bfit, multioutput="raw_values")
    data['csvs'] = csvs
    
    return data

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
        labels = np.array([f"p{p}" for p in peaks])

    return wls, filters, labels


@st.cache
def load_sources(sources_file, wls, thorlabs_list, units=None):
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
    if units is not None:
        sources = unit_conversion(sources, wls, units)
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


def unit_conversion(spectra, wls, units):
    if units == 'E':
        return spectra * STANDARD_FLUX_SCALE
    elif units == 'uE':
        return spectra * (10 ** -6 * STANDARD_FLUX_SCALE)
    elif units == 'W/m^2':
        return dreye.irr2flux(spectra, wls) * STANDARD_FLUX_SCALE
    elif units == 'uW/cm^2':
        return dreye.irr2flux(spectra*100, wls) * STANDARD_FLUX_SCALE
    else:
        raise NameError(f"Unit convention `{units}` not recognized")


@st.cache
def load_spectrum(spectrum_file, wls, units=None):
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
    
    if units is not None:
        spectrum = unit_conversion(spectrum, wls, units)
        
    return spectrum


@st.cache
def load_spectra_samples(spectra_file, wls, units):
    if spectra_file is None:
        return
    df = pd.read_csv(spectra_file).set_index('wls')
    uwls = df.index.to_numpy().astype(np.float64)
    spectra = df.to_numpy().T.astype(np.float64)
    
    if (wls.size != uwls.size) or not np.allclose(wls, uwls):
        spectra = interp1d(
            uwls, spectra, bounds_error=False, fill_value=0, 
            axis=-1
        )(wls)
        
    if units is not None:
        spectra = unit_conversion(spectra, wls, units)

    return spectra


@st.cache
def load_light_source_samples(intensities_samples, units, est):
    df = pd.read_csv(intensities_samples).loc[:, est.sources_labels]
    scalars = dreye.integral(dreye.flux2irr(est.sources), est.wls)
    if units is None:
        return df.to_numpy()
    elif units == 'E':
        return df.to_numpy() * STANDARD_FLUX_SCALE
    elif units == 'uE':
        return df.to_numpy() * (10 ** -6 * STANDARD_FLUX_SCALE)
    elif units == 'W/m^2':
        return df.to_numpy() * scalars
    elif units == 'uW/cm^2':
        return df.to_numpy() * scalars * 100
    else:
        raise NameError(f"Unit convention `{units}` not recognized")
    
    
@st.cache
def compute_hull(est):
    return est.compute_hull()
    
   
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


# initial estimator without sources
@st.cache
def load_initial_receptor_estimator(
    filters, 
    wls, 
    labels, 
):
    est = dreye.ReceptorEstimator(
        filters=filters, 
        domain=wls, 
        labels=labels,
    )
    return est


@st.cache
def get_maxq(est):
    return est.system_relative_capture(est.ub).sum()

### --- START

global_params = load_global_params()


# title of app
st.markdown("## Design custom chromatic stimuli using drEye")

st.markdown(
"""
[drEye](https://dreye.readthedocs.io/en/latest/) is a package that implements various approaches to design stimuli for sensory receptors. 
This web application uses drEye to design chromatic stimuli given an arbitrary set of sensitivities and light sources (e.g. LEDs). 
Since drEye implements a hardware-agnostic approach to stimulus design, different types of light sources
can be combined to design similar stimuli given a set of sensitivities. 

The preprint paper *"Exploiting color space geometry for visual stimulus design across animals"*
available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.01.17.476640v1) explains
key concepts that are implemented in this web application.
"""
)

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
    filters_loaded = n_filters > 1
    initial_est = load_initial_receptor_estimator(
        filters, 
        wls, 
        labels
    )
    
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
        sources_units = None
    else:
        thorlabs_list = None
        sources_file = st.sidebar.file_uploader(
            "Upload a `.csv` file with header columns `wls` and light sources' names.", 
            type='csv', 
            accept_multiple_files=False, 
        )
        sources_units = st.sidebar.selectbox(
            UNITS_TEXT, UNITS_OPTIONS
        )

    if sources_file is None and not thorlabs_list:
        sources_loaded = False
    else:
        sources, sources_labels = load_sources(sources_file, wls, thorlabs_list, units=sources_units)
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
            'intensity spectrum', 
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
        spectrum_units = st.sidebar.selectbox(UNITS_TEXT, UNITS_OPTIONS)
        spectrum = load_spectrum(spectrum_file, wls, units=spectrum_units)
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
    maxq = get_maxq(est)
    hull_size = compute_hull(est)
    estimator_loaded = True
    

# load the various things

@st.cache
def get_filters_colors(filters):
    return sns.color_palette('Greys', len(filters)+2)[1:-1]

@st.cache
def get_sources_colors(sources):
    return sns.color_palette('rainbow', len(sources))

# @st.cache
# def init_chromaticity_state():
#     if 'azimuth' not in st.session_state:
#         st.session_state.azimuth = AZIMUTH_DEFAULT
#     if 'elevation' not in st.session_state:
#         st.session_state.elevation = ELEVATION_DEFAULT
#     if 'playing' not in st.session_state:
#         st.session_state.playing = False
        
def toggle_play():
    st.session_state.playing = not st.session_state.playing
    st.session_state.azimuth -= 1
    
def reset():
    st.session_state.azimuth = AZIMUTH_DEFAULT
    st.session_state.elevation = ELEVATION_DEFAULT
  
@st.cache  
def init_form(est):
    st.session_state.form_submitted = False
    st.session_state.args_submitted = ()    
    

### --- the main window

if not estimator_loaded:
    st.error("REQUIRED: Setup at least 2 opsin sensitivity sets and a set of light sources in the sidebar before proceeding")
    

if filters_loaded and not estimator_loaded:    
    filters_colors = get_filters_colors(filters)
    fig, ax1 = plt.subplots()
    ax1.set_title("Spectral sensitivities")
    fig.set_size_inches(7, 7/2)
    ax1 = initial_est.filter_plot(colors=filters_colors, ax=ax1)
    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('relative sensitivity (a.u.)')
    ax1.legend(loc=2)
    st.pyplot(fig)
    

# with st.expander("Color Model Summary"):
# pairwise plots 
# gamut plot
# capture heat matrix
if estimator_loaded:  
    init_form(est)   
    # st.markdown("### Normalized spectra and spectral sensitivities")
    filters_colors = get_filters_colors(filters)
    fig, ax1 = plt.subplots()
    ax1.set_title("Normalized spectra and spectral sensitivities")
    fig.set_size_inches(7, 7/2)
    ax1 = est.filter_plot(colors=filters_colors, ax=ax1)
    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('relative sensitivity (a.u.)')
    ax1.legend(loc=2)

    sources_colors = get_sources_colors(sources)
    ax2 = plt.twinx(ax1)
    est.sources_plot(colors=sources_colors, ax=ax2)
    ax2.set_ylabel('relative intensity (1/nm)')
    ax2.legend(loc=1)
    st.pyplot(fig)
    
    # set a bunch of points
    # with st.expander("Register and fit captures"):
    st.markdown("### Register and fit captures")
    sampling_method = st.selectbox(
        "Sample creation method", 
        [
            'sample within gamut', 
            'upload spectra',
            'upload intensities', 
            
        ]
    )
    fit_method = st.selectbox(
        "Fit method for samples", 
        [
            "gaussian", 
            "excitation", 
            "poisson", 
            "minimize variance",
            "decompose subframes"
        ],
    )
    
    # form = st.form('Form')
    # with st.form('Form'):
    if sampling_method == 'sample within gamut':
        hull_seed = st.number_input("Seed", min_value=1, value=1)
        hull_samples = st.number_input("Number of samples", min_value=1, value=20)
        hull_l1 = st.number_input(
            "Achromatic value (if 0 all possible achromatic values are sampled)", 
            min_value=0.0, 
            max_value=maxq, 
            value=0.0
        )
        spectra_sample_file = None
        spectra_sample_units = None
        intensity_sample_file = None
        intensity_sample_units = None
    elif sampling_method == 'upload spectra':
        spectra_sample_file = st.file_uploader(
            "Upload a `.csv` file with header columns `wls` and different samples.", 
            type='csv', 
            accept_multiple_files=False,
        )
        spectra_sample_units = st.selectbox(
            UNITS_TEXT, UNITS_OPTIONS
        )
        intensity_sample_file = None
        intensity_sample_units = None
        hull_l1 = None
        hull_samples = None
        hull_seed = None
    elif sampling_method == 'upload intensities':
        intensity_sample_file = st.file_uploader(
            "Upload a `.csv` file with header columns corresponding to the light source names and rows being different samples.", 
            type='csv', 
            accept_multiple_files=False,
        )
        intensity_sample_units = st.selectbox(
            UNITS_TEXT, 
            UNITS_OPTIONS
        )
        spectra_sample_file = None
        spectra_sample_units = None
        hull_l1 = None
        hull_samples = None
        hull_seed = None
    else:
        raise RuntimeError(f"Sampling method not recognized: {sampling_method}")
    
    if sampling_method not in ['sample within gamut']:
        gamut_correction = st.selectbox(
            "Gamut correction", 
            [
                "none", 
                "intensity scaling", 
                "chromatic scaling", 
                "adaptive scaling"
            ]
        )
        
        plot_original = st.checkbox(
            "Plot samples before gamut correction", 
            value=False
        )
    else:
        gamut_correction = "none"
        plot_original = False
        
    if fit_method in ['poisson', 'gaussian', 'excitation']:
        l2_eps = None
        n_layers = None
    elif fit_method == 'minimize variance':
        l2_eps = st.number_input('Allowed L2-error for variance minimization', min_value=0.0001, value=0.01, step=0.001, format="%.4f")
        n_layers = None
    elif fit_method == 'decompose subframes':
        n_layers = st.number_input('Number of subframes (subframes<sources)', min_value=1, max_value=len(sources)-1, value=len(sources)-1)
        l2_eps = None
    else:
        raise RuntimeError(f"Fit method not recognize: {fit_method}")
    
    submitted = st.button('Submit')
        
    if submitted:
        # catching errors
        if sampling_method == 'sample within gamut' and len(sources) < len(filters):
            st.error("System of equations is overdetermined. "
                     "Add more light sources in order to sample in hull. "
                     "There must be more light sources than sensitivities.")
            data = {}
        else:
            st.session_state.form_submitted = True
            st.session_state.args_submitted = (
                sampling_method, gamut_correction, fit_method, 
                hull_samples, hull_seed, hull_l1, 
                spectra_sample_file, spectra_sample_units, 
                intensity_sample_file, intensity_sample_units, 
                l2_eps, n_layers
            )
            data = register_and_fit(
                est, wls, *st.session_state.args_submitted
            )
    elif 'form_submitted' in st.session_state and st.session_state.form_submitted:
        data = register_and_fit(
            est, wls, *st.session_state.args_submitted
        )
    else:
        data = {}
        
    B = data.get('B', None)
    Bfit = data.get('Bfit', None)
    Bbefore = data.get('Bbefore', None)
    
    summaries = f"* Size of the stimulation system relative to a perfect system: {np.round(hull_size, 2)}.\n"
    if 'inhull_before' in data:
        summaries += f"* Fraction of captures within the gamut before correction: {np.round(np.mean(data.get('inhull_before', 0)), 2)}.\n"
    if 'r2before' in data:
        summaries += f"* R2-scores for gamut correction: {np.round(data['r2before'], 2).tolist()}.\n"
    if 'inhull' in data:
        summaries += f"* Fraction of captures within the gamut after correction: {np.round(np.mean(data.get('inhull', 0)), 2)}.\n"
    if 'r2' in data:
        summaries += f"* R2-scores for fit: {np.round(data['r2'], 2).tolist()}.\n"  
    
    # show stats
    st.markdown(
        f"""
### Summary Stats
{summaries}
        """
    )
    
    # download zip file - create cache zip function
    ncols = 3
    cols = st.columns(ncols)
    for idx, (filename, csv) in enumerate(data.get('csvs', {}).items()):
        col = cols[idx % ncols]
        col.download_button(
            f"Download data of {filename.replace('_', ' ')}", 
            csv, 
            file_name=f'{filename}.csv',
            mime='text/csv',
        )
    
    if len(sources) > 1:
        # st.markdown("### Gamut across opsin pairs")
        ncols = (3 if len(filters) > 2 else 1)
        fig, axes = est.gamut_plot(B=B, colors=sources_colors, ncols=ncols, color='gray', alpha=0.5, label='target')
        if plot_original and Bbefore is not None:
            est.gamut_plot(B=Bbefore, sources_vectors=False, ncols=ncols, axes=axes, color='lightgrey', alpha=0.5, label='before\ncorrection', marker='s')
        if Bfit is not None:
            est.gamut_plot(B=Bfit, sources_vectors=False, ncols=ncols, axes=axes, color='black', alpha=0.5, label='fits', marker='x')
        fig.suptitle("Gamut across opsin pairs")
        # create legend
        handles, labels = axes[0].get_legend_handles_labels()
        
        if ncols == 3:
            fig.legend(handles, labels, bbox_to_anchor=(1.2, 0.7))
            fig.set_size_inches(15/2, 5/2 * np.ceil(len(axes) / 3))
            st.pyplot(fig)
        else:
            fig.legend(handles, labels, bbox_to_anchor=(1.5, 0.7))
            fig.set_size_inches(5/2, 5/2)
            col1, col2 = st.columns(2)
            col1.pyplot(fig)
        
    if len(filters) in [2, 3] and (len(filters) <= len(sources)):
        # st.markdown("### Chromaticity diagram")
        if len(filters) == 2:
            col = col2
        else:
            _, col, _ = st.columns([1, 3, 1])
        fig, ax = plt.subplots()
        fig.set_size_inches(10/2, 7/2)
        ax.set_title("Chromaticity diagram")
        est.simplex_plot(B=B, ax=ax, color='gray', alpha=0.5)
        col.pyplot(fig)
        
    if len(filters) in [4] and (len(filters) <= len(sources)):
        # st.markdown("### Chromaticity diagram")
        _, col, _ = st.columns([1, 3, 1])
        
        if 'azimuth' not in st.session_state:
            st.session_state.azimuth = AZIMUTH_DEFAULT
        if 'elevation' not in st.session_state:
            st.session_state.elevation = ELEVATION_DEFAULT
        if 'playing' not in st.session_state:
            st.session_state.playing = False
        # elevation = st.slider("Elevation", min_value=-90, max_value=90, value=30, step=15)
        # azimuth = st.slider("Azimuth", min_value=-180, max_value=180, value=-45, step=15)
        ax = est.simplex_plot(B=B, color='black', alpha=1)
        ax.view_init(ELEVATION_DEFAULT, st.session_state.azimuth)
        fig = plt.gcf()
        fig.set_size_inches(7/2, 7/2)
        ax.set_title("Chromaticity diagram")
        
        player = col.pyplot(fig)
        
        # controls
        _, _, lcenter, center, _, _ = st.columns(6)
        lcenter.button('Reset', on_click=reset)
        center.button('Play/Pause', on_click=toggle_play)
        
        while st.session_state.playing:
            st.session_state.azimuth = (st.session_state.azimuth + 1)
            ax.view_init(ELEVATION_DEFAULT, st.session_state.azimuth)
            player.pyplot(fig)