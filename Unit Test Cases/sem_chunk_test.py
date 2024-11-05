import unittest
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


class TestSemanticChunking(unittest.TestCase):

    def test_pre_process_data(self):
        conv = "spk_0: Hello\nspk_1: Hi there!\nspk_0: How are you?"
        expected_output = ["Hello", "Hi there!", "How are you?"]
        self.assertEqual(pre_process_data(conv), expected_output)

    def test_combine_sentences_non_overlap(self):
        sentences = [{'sentence': 'This is sentence ' + str(i), 'index': i} for i in range(10)]
        combined_sentences = combine_sentences_non_overlap(sentences, buffer_size=2)
        self.assertEqual(len(combined_sentences), 10)
        self.assertEqual(combined_sentences[2]['combined_sentence_previous'], "This is sentence 0 This is sentence 1 ")
        self.assertEqual(combined_sentences[2]['combined_sentence_next'], "This is sentence 2 This is sentence 3 ")

    def test_get_sent_transformer_embeddings(self):
        # Mocking the SentenceTransformer model to avoid actual embedding generation
        class MockSentenceTransformer:
            def encode(self, sentences, batch_size, show_progress_bar, convert_to_tensor):
                return np.random.rand(len(sentences), 384)  # Example embedding size

        global sent_model  # Accessing the global sent_model variable
        sent_model = MockSentenceTransformer()

        comb_sent_prev = ["This is a test sentence.", "Another test sentence."]
        comb_sent_next = ["This is the next sentence.", "And another one."]

        embeddings_prev, embeddings_next = get_sent_transformer_embeddings(comb_sent_prev, comb_sent_next)
        self.assertEqual(embeddings_prev.shape, (2, 384))
        self.assertEqual(embeddings_next.shape, (2, 384))

    def test_normalize_data(self):
        embeddings_prev = np.random.rand(5, 10)
        embeddings_next = np.random.rand(5, 10)

        norm_emb_prev, norm_emb_next = normalize_data(embeddings_prev, embeddings_next)

        # Check if the normalized vectors have L2 norm close to 1
        for i in range(5):
            self.assertAlmostEqual(np.linalg.norm(norm_emb_prev[i]), 1.0, places=5)
            self.assertAlmostEqual(np.linalg.norm(norm_emb_next[i]), 1.0, places=5)

    def test_calculate_cosine_distance(self):
        sentences = [{'sentence': 'Sentence ' + str(i), 'index': i} for i in range(3)]
        norm_emb_prev = np.array([[1, 0], [0.707, 0.707], [0, 1]])
        norm_emb_next = np.array([[0.707, 0.707], [0, 1], [1, 0]])

        distances, updated_sentences = calculate_cosine_distance(sentences, norm_emb_prev, norm_emb_next)

        expected_distances = [1 - (1 * 0.707 + 0 * 0.707), 1 - (0.707 * 0 + 0.707 * 1), 1 - (0 * 1 + 1 * 0)]
        self.assertListAlmostEqual(distances, expected_distances, places=3)
        for i in range(3):
            self.assertEqual(updated_sentences[i]['distance_to_next'], distances[i])

    def test_find_peak_values(self):
        distances = [0.1, 0.2, 0.8, 0.3, 0.7, 0.2, 0.9, 0.1]
        threshold = 0.5
        expected_peak_indices = [2, 4, 6]  # Indices of 0.8, 0.7, and 0.9
        peak_indices = find_peak_values(threshold, distances)
        self.assertEqual(list(peak_indices), expected_peak_indices)

    def test_merge_small_chunks(self):
        processed_data = ["This is", "a test", "sentence.", "Another", "test", "sentence."]
        indices = [0, 2, 4]  # Initial split points
        merged_indices = merge_small_chunks(processed_data, indices)
        # Assuming LLM_MAX_TOKENS is large enough to merge all
        self.assertEqual(merged_indices, [0, 4])  

    def test_calculate_chunk_sizes(self):
        processed_data = ["This is", "a test", "sentence.", "Another", "test", "sentence."]
        split_boundries_arr = [1, 3]
        chunk_sizes = calculate_chunk_sizes(processed_data, split_boundries_arr)
        expected_sizes = [3, 3, 2]  # ["This is a", "test sentence. Another", "test sentence."]
        self.assertEqual(list(chunk_sizes), expected_sizes)

    def test_all_chunks_big(self):
        # ... (Requires setting up processed_data and distances) ...
        pass  # Implement a test case with specific data

    def test_optimized_peak_values(self):
        # ... (Requires setting up processed_data and distances) ...
        pass  # Implement a test case with specific data

    def test_prepare_text_chunks(self):
        # ... (Requires setting up processed_data and distances) ...
        pass  # Implement a test case with specific data

    def test_invoke_semantic_chunking(self):
        # ... (Requires setting up a conv string) ...
        pass  # Implement a test case with specific data

    def assertListAlmostEqual(self, list1, list2, places):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places)

if __name__ == '__main__':
    unittest.main()
