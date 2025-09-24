import os
from sklearn.preprocessing import normalize
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
# import gpt_ada_embedding_api
from typing import List

import re
import time
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

#Declaring constants
BUFFER_SIZE = 25
LLM_MAX_TOKENS = 8000
THRESHOLD_DECREASING_SPEED = 0.1

# process data
def pre_process_data(conv: str):
    """
    Preprocesses the conversation string by removing speaker identifiers.

    Args:
        conv: The conversation string containing speaker identifiers (e.g., "spk_0: Hello").

    Returns:
        A list of processed sentences without speaker identifiers.
    """
    
    processed_data = []
    for phrase in conv.split('\n'):
        processed_data.append(re.sub(r'((spk_0)|(spk_1)):\s+','',phrase))
    return processed_data


#combine sentence non-overlap
def combine_sentences_non_overlap(sentences,buffer_size=5):
    """
    Combines sentences within a buffer range to create context for each sentence.

    Args:
        sentences: A list of dictionaries, each containing a sentence and its index.
        buffer_size: The number of sentences to consider before and after each sentence.

    Returns:
        A list of dictionaries with added keys for combined previous and next sentences.
    """
    
    for i in range(len(sentences)):
        combined_sent = ""

        if i >=buffer_size and i <= (len(sentences) - buffer_size):
            #add previous sentences
            previous_sent = ''
            next_sent = ''

            for j in range(i-buffer_size,i):
                previous_sent += sentences[j]['sentence']+ ' '

            sentences[i]['combined_sentence_previous'] = previous_sent

            for k in range(i,i + buffer_size):
                next_sent += sentences[k]['sentence'] + ' '
            
            sentences[i]['combined_sentence_next'] = next_sent
        else:
            sentences[i]['combined_sentence_previous'] = ""
            sentences[i]['combined_sentence_next'] = ""

    return sentences

USE_SENTENCE_TRANSFORMER= True
if USE_SENTENCE_TRANSFORMER:
    try:
        get_ipython().system('pip install sentence-transformers --q')
    except NameError:
        subprocess.check_call(['pip', 'install', 'sentence-transformers', '--quiet'])

    from sentence_transformers import SentenceTransformer, util

    sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('SentenceTransformer model loaded')


def get_sent_transformer_embeddings(comb_sent_prev,comb_sent_next):
    """
    Generates sentence embeddings for previous and next context sentences using SentenceTransformer.

    Args:
        comb_sent_prev: A list of combined previous sentences.
        comb_sent_next: A list of combined next sentences.

    Returns:
        A tuple containing numpy arrays of embeddings for previous and next sentences.
    """
    
    embeddings_prev = sent_model.encode(comb_sent_prev,batch_size=128, show_progress_bar=True, convert_to_tensor=True)
    print(embeddings_prev.shape)

    embeddings_next = sent_model.encode(comb_sent_next,batch_size=128, show_progress_bar=True, convert_to_tensor=True)
    print(embeddings_prev.shape)

    embeddings_prev = embeddings_prev.cpu().numpy()
    embeddings_next = embeddings_next.cpu().numpy()

    return embeddings_prev, embeddings_next

def normalize_data(embeddings_prev, embeddings_next):
    """
    Normalizes the embeddings using L2 normalization.

    Args:
        embeddings_prev: Embeddings for previous sentences.
        embeddings_next: Embeddings for next sentences.

    Returns:
        A tuple containing normalized embeddings for previous and next sentences.
    """
    
    norm_emb_prev = normalize(embeddings_prev, norm='L2')
    norm_emb_next = normalize(embeddings_next,norm='L2')

    return norm_emb_prev, norm_emb_next


# perform cosine similarity check
def calculate_cosine_distance(sentences, norm_emb_prev, norm_emb_next):
    """
    Calculates the cosine distance between normalized embeddings of previous and next sentences.

    Args:
        sentences: A list of dictionaries containing sentence information.
        norm_emb_prev: Normalized embeddings for previous sentences.
        norm_emb_next: Normalized embeddings for next sentences.

    Returns:
        A tuple containing a list of cosine distances and the updated list of sentences 
        with distance information.
    """
    
    distances = []
    for i in range(len(norm_emb_prev)):

        #calculate cosine similarity
        similarity = cosine_similarity([norm_emb_prev[i]],[norm_emb_next[i]])[0][0]

        #convert to cosine distance
        distance = 1 -  similarity

        #append cosine distance to the list
        distances.append(distance)

        # store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    return distances, sentences


def find_peak_values(threshold:float, distances: List[float]):
    """
    Identifies peak values in the cosine distances that are above a given threshold.

    Args:
        threshold: The threshold value for identifying peaks.
        distances: A list of cosine distances.

    Returns:
        A list of indices representing the peak values in the distances.
    """
    
    vals = [0 if dist < threshold else dist for dist in distances]

    # no value above threshold then return an empty array
    if (np.array(vals) == 0).all():
        return np.array([])
    
    final_peak_value = []
    for i in range(len(vals)-1):
        if vals[i] != vals[i+1]:
            final_peak_values.append(vals[i])

            if i == len(vals) - 2:
                final_peak_values.append(vals[i+1])
    
    final_peak_values = np.array(final_peak_values)

    split_points = np.where(final_peak_values == 0.)[0]

    print('find_peak_values_split-->', np.split(final_peak_values, split_points))
    peak_vals = [peaks.max() for peaks in np.split(final_peak_values, split_points) if len(peaks) != 0]
    dist_series = pd.Series(distances)

    return dist_series[dist_series.isin(peak_vals)].index


def merge_small_chunks(processed_data, indices):
    """
    Merges small chunks of text to ensure they meet the minimum token size requirement.

    Args:
        processed_data: A list of processed sentences.
        indices: A list of indices representing potential split points.

    Returns:
        A list of indices representing the final split points after merging small chunks.
    """
    
    m_chunk_arr = [0]
    win_arr = np.lib.stride_tricks.sliding_window_view(np.array(indices), 2, axis=0)

    for i in range(0,len(win_arr)):
        if win_arr[i][-1] not in m_chunk_arr:
            if i == len(win_arr) -1:
                m_chunk_arr.append(win_arr[i][-1])
                break

            if len((' '.join(processed_data[win_arr[i][0]: win_arr[i+1][-1]]).split())) < LLM_MAX_TOKENS:
                m_chunk_arr.append(win_arr[i+1][-1])
            else:
                m_chunk_arr.append(win_arr[i][-1])

    if m_chunk_arr == indices:
        return m_chunk_arr
    
    return merge_small_chunks(processed_data, m_chunk_arr)



def calculate_chunk_sizes(processed_data: List[str], split_boundries_arr: List[int]) -> List[int]:
    """
    Calculates the size of each chunk in terms of number of words.

    Args:
        processed_data: A list of processed sentences.
        split_boundries_arr: A list of indices representing the split points.

    Returns:
        A list of chunk sizes.
    """
    
    adjusted_buffer_boundries = split_boundries_arr + BUFFER_SIZE
    print('split_boundries_arr', split_boundries_arr)

    #append start and end to the array
    adjusted_buffer_boundries = [0] + list(adjusted_buffer_boundries) + [len(processed_data)]
    print('adjusted_buffer_boundries',adjusted_buffer_boundries)

    chunk_length_arr = []
    for index in range(len(adjusted_buffer_boundries) -1):
        chunk_len = (" ".join(processed_data[adjusted_buffer_boundries[index]: adjusted_buffer_boundries[index + 1]])).split().__len__()
        chunk_length_arr.append(chunk_len)
    print('chunk_len_arr',chunk_length_arr)

    return np.array(chunk_length_arr)


def all_chunks_big(processed_data,threshold, distances):
    """
    Checks if all chunks after splitting with the given threshold are larger than the 
    maximum allowed token size.

    Args:
        processed_data: A list of processed sentences.
        threshold: The threshold value for identifying peaks in cosine distances.
        distances: A list of cosine distances.

    Returns:
        True if all chunks are larger than the maximum token size, False otherwise.
    """
    
    split_borders_indices = find_peak_values(threshold, distances)

    if len(split_borders_indices) == 0:
        return True
    
    chunk_sizes = calculate_chunk_sizes(processed_data, split_borders_indices)
    if (chunk_sizes < LLM_MAX_TOKENS).all():
        return False
    else:
        return True


def optimized_peak_values(processed_data, distances):
    """
    Finds the optimal peak values in cosine distances by iteratively decreasing the 
    threshold until all chunks are smaller than the maximum token size.

    Args:
        processed_data: A list of processed sentences.
        distances: A list of cosine distances.

    Returns:
        A list of indices representing the final split points.
    """
    
    threshold = 1.0
    while all_chunks_big(processed_data, threshold, distances):
        threshold -= 0.01* THRESHOLD_DECREASING_SPEED
    
    print('*'*30)
    peak_indices = find_peak_values(threshold, distances)
    print('final threshold-->', threshold)
    print('FINAL PEAK INDICES', peak_indices)

    return [0]+list(peak_indices+ BUFFER_SIZE) + [len(processed_data)]


def prepare_text_chunks(processed_data, distances):
    """
    Prepares the final text chunks for dispatch to the language model.

    Args:
        processed_data: A list of processed sentences.
        distances: A list of cosine distances.

    Returns:
        A list of text chunks, where each chunk is a list of sentences.
    """
    
    opv = optimized_peak_values(processed_data, distances)
    pv = merge_small_chunks(processed_data, opv)
    print(pv)
    data_chunks = [processed_data[pv[i]:pv[i+1]] for i in range(len(pv)-1)]
    print('chunk sizes at dispatch', [len((" ".join(chunk_arr)).split()) for chunk_arr in data_chunks])

    return data_chunks


def invoke_semantic_chunking(conv: str) -> List[str]:
    """
    Performs semantic chunking on a conversation string.

    This function preprocesses the conversation, generates sentence embeddings,
    calculates cosine distances between sentences, and identifies optimal split points
    to create chunks suitable for processing by a language model.

    Args:
        conv: The conversation string.

    Returns:
        A list of text chunks, where each chunk is a list of sentences.
    """
    
    processed_data = pre_process_data(conv)

    sentences = [{'sentence':x,'index':i} for i,x in enumerate(processed_data)]
    sentences = combine_sentences_non_overlap(sentences, BUFFER_SIZE)

    comb_sent_prev = [dict['combined_sentence_previous'] for dic in sentences]
    comb_sent_next = [dict['combined_sentence_next'] for dic in sentences]

    print(comb_sent_prev.__len__(), comb_sent_next.__len__())
    embeddings_prev, embeddings_next = get_sent_transformer_embeddings(comb_sent_prev, comb_sent_next)
    norm_emb_prev, norm_emb_next = normalize_data(embeddings_prev, embeddings_next)
    distances, sentences = calculate_cosine_distance(sentences, norm_emb_prev, norm_emb_next)

    return prepare_text_chunks(processed_data, distances)
