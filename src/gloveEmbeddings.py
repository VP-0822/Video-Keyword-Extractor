import numpy as np
import embeddingInterface
import config

class GloveEmbedding(embeddingInterface.EmbeddingInterface):
    def __init__(self, embedding_vector_file, embedding_dimension):
        self.embedding_vector_file = embedding_vector_file
        self.embedding_dimension = embedding_dimension
        self.load()
    
    def load(self):
        vector_file = open(self.embedding_vector_file, encoding="utf-8")
        self.embeddings_index = dict()
        for wordline in vector_file:
            vector_values = wordline.split()
            # First element in values is always actual word for e.g. 'Soccer'
            word = vector_values[0]
            vector_coeffs = np.asarray(vector_values[1:], dtype="float32")
            self.embeddings_index[word] = vector_coeffs
        vector_file.close()

    def getEmbeddingVectorFor(self, word_list, vocab_size):
        self.select_word_embedding_matrix = np.zeros((vocab_size, self.embedding_dimension))
        for index, word in enumerate(word_list):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.select_word_embedding_matrix[index] = embedding_vector
            else:
                # Word not present in embedding vector list, embeddding will be all zeros
                pass
        
        return self.select_word_embedding_matrix

if __name__ == "__main__":
    ge = GloveEmbedding(config.GLOVE_200_DIM_FILE, 200)   
    print(len(ge.embeddings_index))
    ge_embedd = ge.embeddings_index.get('woman')
    my_embedd = ge.getEmbeddingVectorFor(['woman'], 1)
    start_embedd = ge.embeddings_index.get('>')
    stop_embedd = ge.embeddings_index.get('<')
    none_embedd = ge.embeddings_index.get('?!?')
    extra_embedd = ge.embeddings_index.get('___')
    print(ge_embedd)
    print(my_embedd) 
    print('=======================')
    print(start_embedd)
    print('=======================')
    print(stop_embedd)
    print('=======================')
    print(none_embedd)
    print('=======================')
    print(extra_embedd)
