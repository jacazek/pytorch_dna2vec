# pytorch_dna2vec
Train an embedding on a given set of FASTA files. Requires vocabulary compatible with the kmer size chosen for the embedding.

## Questions
1. Would kmers actually represent the vocabulary used to build genes? Like words are the vocabulary of language?


## Ways to verify the embedding
1. Analogical reasoning - addition and subtraction of certain embeddings maps to analogy
   1. embedding('king') - embedding('man') + embedding('woman') = embedding('queen')
   2. The result should probably be compared to k-nearest neighbors
   3. How would this translate to DNA? Does it even make sense at a kmer level?
2. Synonym/Antonym detection
   1. cosine_similarity(embedding('happy'), embedding('joyful')) > cosine_similarity(embedding('happy'), embedding('sad'))
   2. Again, how would this translate to DNA and does kmer level comparison make sense
3. Clustering similar words
   1. Words with similar meaning are near each other (k-nearest?)
4. Semantic composition
   1. Sums of two embeddings should approximate their compound.  The vocabulary would need to be able to support compounts
   2. Would need a variable kmer strategy
5. Dimensionality reduction - read more on this.
6. Contextual similarity  for contextual embeddings
   1. Cosine similary of the same word is within a threshold given two different contexts
7. Downstream task performance
   1. Gene recognition does well
8. Lexical semantics
   1. cosine_similarity(embedding('animal'), embedding('dog')) > cosine_similarity(embedding('animal'), embedding('car'))
9. Translation invariance - compare embeddings of translated word pairs across languges (genes?, chromosomes?)
   10. 