from .search_utils import quick_search_sorted
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from .spectral_denoising import spectral_denoising_with_master_formulas, electronic_denoising, prep_formula, has_benzene
from .spectral_operations import entropy_similarity, sort_spectrum
from .file_io import standardize_col
import pandas as pd

pd.options.mode.chained_assignment = None


def denoising_search_batch(msms_query, pmz_query, reference_lib, identitiy_search_mass_error=0.01, mass_tolernace=0.005,
                           pmz_col='precursor_mz',
                           smiles_col='smiles',
                           adduct_col='adduct', msms_col='peaks', first_n='all'):
    """
    Perform batch denoising search on given MS/MS data and precursor m/z values with parallel processing.

    Parameters:
        msms_query (list): List of MS/MS spectra to be denoised.

        pmz_query (list): List of precursor m/z values corresponding to the MS/MS spectra.

        reference_lib (pandas.DataFrame): Reference library containing known spectra for comparison.

        identitiy_search_mass_error (float, optional): Mass error tolerance for identity search. Default is 0.01.

        mass_tolerance (float, optional): Maximum allowed tolerance for denoising. Default is 0.005.

        pmz_col (str, optional): Column name for precursor m/z in the reference library. Default is 'precursor_mz'.

        smiles_col (str, optional): Column name for SMILES in the reference library. Default is 'smiles'.

        adduct_col (str, optional): Column name for adducts in the reference library. Default is 'adduct'.

        msms_col (str, optional): Column name for MS/MS peaks in the reference library. Default is 'peaks'.
    Returns:
        pandas.DataFrame: DataFrame containing the results of the denoising search. Each index in the result DataFrame corresponds to the denoising search result of the corresponding input MS/MS spectrum.
    """
    # print('im newnew')
    # reference_lib.reset_index(drop=True, inplace=True)
    reference_lib.sort_values(by=pmz_col, inplace=True, ascending=True)
    reference_lib.dropna(subset=[smiles_col], inplace=True)
    if len(msms_query) != len(pmz_query):
        print('The length of msms and pmz should be the same')
        return pd.DataFrame()

    with mp.Pool(processes=6) as pool:
        # Use starmap to handle multiple parameters
        results = pool.starmap(denoising_search,
                               tqdm([(msms, pmz, reference_lib, identitiy_search_mass_error, mass_tolernace,
                                      pmz_col, smiles_col, adduct_col, msms_col, first_n, False) for msms, pmz in
                                     zip(msms_query, pmz_query)], total=len(pmz_query)))

    return results


def denoising_search(msms, pmz, reference_lib, identitiy_search_mass_error=0.01, mass_tolernace=0.005,
                     pmz_col='precursor_mz', smiles_col='smiles', adduct_col='adduct', msms_col='peaks',
                     first_n=1, need_sort=True):
    if need_sort:
        reference_lib = reference_lib.sort_values(by=pmz_col, ascending=True).copy()
    # reference_lib.reset_index(drop=True, inplace=True)
    pmz_candidates = quick_search_sorted(reference_lib, pmz_col, pmz - identitiy_search_mass_error,
                                         pmz + identitiy_search_mass_error)
    if pmz_candidates.empty:
        return pd.DataFrame()
    pmz_candidates, unique_formulas, benzene_tag = get_all_master_formulas(pmz_candidates, smiles_col, adduct_col)
    msms_raw = msms
    msms = electronic_denoising(msms)

    denoised_spectra_map = {}
    for i, formula in enumerate(unique_formulas):
        # Store the result, which might be np.nan or an array
        denoised_spectra_map[formula] = spectral_denoising_with_master_formulas(
            msms, formula, benzene_tag[i], pmz, mass_tolernace
        )

    entropy_similarities = []
    denoised_similarities = []
    final_denoised_peaks_list = []

    for row in pmz_candidates.itertuples(index=False):
        ref_peaks = getattr(row, msms_col)  # Access column data via attribute
        master_formula = getattr(row, 'master_formula')

        denoised_peaks_candidate = denoised_spectra_map.get(master_formula)

        is_denoising_valid = isinstance(denoised_peaks_candidate, np.ndarray) and denoised_peaks_candidate.size > 0

        spectrum_for_denoised_calc = denoised_peaks_candidate if is_denoising_valid else msms
        final_denoised_peaks_for_row = denoised_peaks_candidate if is_denoising_valid else msms  # Store the actual peaks used

        entropy_sim = entropy_similarity(msms_raw, ref_peaks, pmz=pmz)
        denoised_sim = entropy_similarity(spectrum_for_denoised_calc, ref_peaks, pmz=pmz)

        entropy_similarities.append(entropy_sim)
        denoised_similarities.append(denoised_sim)
        final_denoised_peaks_list.append(final_denoised_peaks_for_row)

    pmz_candidates['entropy_similarity'] = entropy_similarities
    pmz_candidates['denoised_similarity'] = denoised_similarities
    pmz_candidates['denoised_peaks'] = final_denoised_peaks_list
    pmz_candidates['query_peaks'] = [msms_raw] * len(pmz_candidates)
    pmz_candidates['query_pmz'] = pmz

    pmz_candidates.sort_values(by='denoised_similarity', ascending=False, inplace=True)
    if isinstance(first_n, int):
        return pmz_candidates.head(first_n)
    elif first_n == 'all':
        return pmz_candidates
    else:
        print('please specify the number of results to return')
        return np.nan
    # return pmz_candidates.head(first_n)

    # return pmz_candidates.head(first_n)


def get_all_master_formulas(pmz_candidates, smiles_col='smiles', adduct_col='adduct'):
    smiles_array = pmz_candidates[smiles_col].values
    adduct_array = pmz_candidates[adduct_col].values
    n = len(smiles_array)

    master_formulas = [None] * n
    benzene_tag = [None] * n

    for i, (smiles, adduct) in enumerate(zip(smiles_array, adduct_array)):
        try:
            master_formulas[i] = prep_formula(smiles, adduct)
            benzene_tag[i] = has_benzene(smiles)
        except:
            master_formulas[i] = np.nan
            benzene_tag[i] = np.nan

    result_df = pmz_candidates.copy()
    result_df['master_formula'] = master_formulas

    benzene_tag_np = np.array(benzene_tag)

    formula_indices = {}
    for i, formula in enumerate(master_formulas):
        if formula == formula:
            if formula not in formula_indices:
                formula_indices[formula] = []
            formula_indices[formula].append(i)

    unique_formulas = list(formula_indices.keys())

    unique_formulas_benzene_tags = []
    for formula in unique_formulas:
        indices = formula_indices[formula]
        unique_formulas_benzene_tags.append(benzene_tag_np[indices].sum() > 0)

    return result_df, unique_formulas, unique_formulas_benzene_tags

# def denoising_search(msms, pmz, reference_lib, identitiy_search_mass_error=0.01, mass_tolernace = 0.005,
#                      pmz_col = 'precursor_mz', smiles_col = 'smiles', adduct_col = 'adduct', msms_col = 'peaks'):
#     """
#     Perform a denoising search on mass spectrometry data.
#     In addition to traditional identity search, the query msms will be denoised using molecular information of all candidates within predefined identitiy_search_mass_error range.
#     Then, entropy similarity scores are calculated for both the original and denoised spectra.

#     Parameters:
#         msms (np.array): The mass spectrometry data (peaks) to be denoised.

#         pmz (float): The precursor m/z value associated with the query MS/MS spectra.

#         reference_lib (pandas.DataFrame): The reference library containing known spectra and associated metadata.

#         identitiy_search_mass_error (float, optional): The mass error tolerance for the identity search. Default is 0.01.

#         mass_tolernace (float, optional): The mass tolerance for spectral denoising. Default is 0.005.

#         pmz_col (str, optional): The column name for precursor m/z values in the reference library. Default is 'precursor_mz'.

#         smiles_col (str, optional): The column name for SMILES strings in the reference library. Default is 'smiles'.

#         adduct_col (str, optional): The column name for adduct information in the reference library. Default is 'adduct'.

#         msms_col (str, optional): The column name for MS/MS peaks in the reference library. Default is 'peaks'.
#     Returns:
#         pd.DataFrame: A DataFrame containing all the candidate spectra, with entropy similarity scores calculated based on both denoised and raw spectra.
#     """
#     pmz_candidates = quick_search_values(reference_lib, pmz_col, pmz - identitiy_search_mass_error, pmz + identitiy_search_mass_error)

#     if len(pmz_candidates) == 0:
#         return pd.DataFrame()#no pmz matches
#     for index, row in pmz_candidates.iterrows():
#         try:
#             msms_d = spectral_denoising(msms, row[smiles_col], row[adduct_col], mass_tolerance=mass_tolernace)

#         except:
#             msms_d = np.nan
#             # continue
#         pmz_candidates.loc[index, 'entropy_similarity'] = entropy_similairty(msms, row[msms_col], pmz = pmz, ms2_error = identitiy_search_mass_error)
#         pmz_candidates.loc[index, 'denoised_similarity'] = entropy_similairty(msms_d, row[msms_col], pmz = pmz, ms2_error = identitiy_search_mass_error)
#     pmz_candidates['query_peaks']=[msms]*len(pmz_candidates)
#     pmz_candidates['query_peaks_denoised']=[msms_d]*len(pmz_candidates)
#     pmz_candidates['query_pmz']=pmz
#     pmz_candidates.sort_values(by='denoised_similarity', ascending=False, inplace=True)
#     # cols = ['name', 'cas','adduct','smiles','query_pmz', 'query_peaks','query_peaks_denoised','entropy_similarity','denoised_similarity']
#     # df_return = pmz_candidates[cols]

#     # df_return.sort_values(by='denoised_similarity', ascending=False, inplace=True)
#     pmz_candidates = standardize_col(pmz_candidates)
#     return pmz_candidates
