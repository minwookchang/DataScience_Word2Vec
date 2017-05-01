#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define SIG_TABLE_SIZE 1000
#define MAX_SIG 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_HUFF_LENGTH 40

const int dict_hash_size = 30000000;

typedef float real;

struct vocab_word {
	long long cn;
	int *huff_path;
	char *word, *huff_code, huff_length;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *dict;

int cbow = 0, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *dict_hash;
long long dict_max_size = 1000, dict_size = 0, hidden_size = 300;
long long trained_words = 0, word_count_actual = 0, epochs = 5, file_size= 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

real *ih_weight, *ho_weight_hs, *ho_weigth_neg, *sig_table;
clock_t start;

int hs = 0, negative = 5;
const int uni_table_size = 1e8;
int *uni_table;

/*
 * This function is for negative sampling.
 * By using Unigram Distribution, fill the table with words index.
 */
void initUnigramTable() {
	int a, i;
	double power_total = 0;
	double d1, power = 0.75;
	uni_table = (int *)malloc(uni_table_size * sizeof(int));
	for(a = 0; a < dict_size; a++) power_total += pow(dict[a].cn, power);
	i = 0;
	d1 = pow(dict[i].cn, power) / power_total;
	for(a = 0; a < uni_table_size; a++) {
		uni_table[a] = i;
		if(a / (double)uni_table_size > d1) {
			i++;
			d1 += pow(dict[i].cn, power) / power_total;
		}
		if (i >= dict_size) i = dict_size - 1;
	}
}

/*
 * Read word assuming that space, tab, eol is the boundary of words
 */
void readWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if(ch == 13) continue;

		if((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if(ch == '\n') ungetc(ch, fin);
				break;
			}

			if(ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		if(a >= MAX_STRING - 1) a--;
	}
	word[a] = 0;
}

/*
 * Using djb2 hash
 */
int getWordHash(char *word) {
	unsigned long long a, hash = 0;
	for(a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % dict_hash_size;
	return hash;
}

/*
 * Return index of word in dict, if none then return -1
 */
int searchVocab(char *word) {
	unsigned int hash = getWordHash(word);

	while(1) {
		if(dict_hash[hash] == -1) return -1;
		if(!strcmp(word, dict[dict_hash[hash]].word)) return dict_hash[hash];
		hash = (hash + 1) % dict_hash_size;
	}
	return -1;
}

/*
 * This is used after dict is all set.
 * When training cbow or skip-gram, read word from file and return its index
 */
int readWordIndex(FILE *fin) {
	char word[MAX_STRING];
	readWord(word, fin);
	if(feof(fin)) return -1;
	return searchVocab(word);
}

int addWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;

	if(length > MAX_STRING) length = MAX_STRING;
	dict[dict_size].word = (char *)calloc(length, sizeof(char));
	strcpy(dict[dict_size].word, word);

	dict[dict_size].cn = 0;
	dict_size++;

	if(dict_size + 2 > dict_max_size) {
		dict_max_size += 1000;
		dict = (struct vocab_word *)realloc(dict, dict_max_size * sizeof(struct vocab_word));
	}
	hash = getWordHash(word);
	while(dict_hash[hash] != -1)
		hash = (hash + 1) % dict_hash_size;

	dict_hash[hash] = dict_size - 1;
	return dict_size - 1;
}

int vocabCompare(const void *a, const void *b) {
	return((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void sortVocab() {
	int a, size;
	unsigned int hash;

	//sort without first "</s>"
	qsort(&dict[1], dict_size - 1, sizeof(struct vocab_word), vocabCompare);

	for(a = 0; a < dict_hash_size; a++) dict_hash[a] = -1;
	size = dict_size;
	trained_words = 0;

	for(a = 0; a < size; a++) {
		if((dict[a].cn < min_count) && (a != 0)) {
			dict_size--;
			free(dict[a].word);
		} else {
			hash = getWordHash(dict[a].word);
			while(dict_hash[hash] != -1) hash = (hash + 1) % dict_hash_size;
			dict_hash[hash] = a;
			trained_words += dict[a].cn;
		}
	}

	dict = (struct vocab_word *)realloc(dict, (dict_size + 1) * sizeof(struct vocab_word));
	for(a = 0; a < dict_size; a++) {
		dict[a].huff_code = (char *)calloc(MAX_HUFF_LENGTH, sizeof(char));
		dict[a].huff_path = (int *)calloc(MAX_HUFF_LENGTH, sizeof(int));
	}

void reduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for(a = 0; a < dict_size; a++) if(dict[a].cn > min_reduce) {
		dict[b].cn = dict[a].cn;
		dict[b].word = dict[a].word;
		b++;
	} else free(dict[a].word);
	dict_size = b;
	for(a = 0; a < dict_hash_size; a++) dict_hash[a] = -1;
	for(a = 0; a < dict_size; a++) {
		hash = getWordHash(dict[a].word);
		while(dict_hash[hash] != -1) hash = (hash + 1) % dict_hash_size;
		dict_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

void assignHuffTree()
{
	long long a, b, i, min1i, min2i, pos1, pos2, path[MAX_HUFF_LENGTH];
	char code[MAX_HUFF_LENGTH];
	long long *count = (long long *)calloc(dict_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(dict_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(dict_size * 2 + 1, sizeof(long long));
	//fill with current word counts
	for(a = 0;a < dict_size;a++) count[a] = dict[a].cn;
	
	//fill middle node section with large number
	for(a = dict_size;a < dict_size * 2;a++) count[a] = 1e15;
	pos1 = dict_size - 1;
	pos2 = dict_size;

	for(a = 0; a < dict_size; a++)
	{
		//find first min index
		if(pos1 >= 0)
		{
			if(count[pos1] < count[pos2])
			{
				min1i = pos1;
				pos1--;
			} else
			{
				min1i = pos2;
				pos2++;
			}
		} else
		{
			min1i = pos2;
			pos2++;
		}
		//find second min index
		if(pos1 >= 0)
		{
			if(count[pos1] < count[pos2])
			{
				min2i = pos1;
				pos1--;
			} else
			{
				min2i = pos2;
				pos2++;
			}
		} else
		{
			min2i = pos2;
			pos2++;
		}

		count[dict_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = dict_size + a;
		parent_node[min2i] = dict_size + a;
		binary[min2i] = 1;
	}

	for(a = 0; a < dict_size; a++)
	{
		b = a;
		i = 0;
		while (1)
		{
			code[i] = binary[b];
			path[i] = b;
			i++;
			b = parent_node[b];
			//if parent is root then break
			if (b == dict_size * 2 - 2) break;
		}
		dict[a].huff_length = i;
		//first path is root
		dict[a].huff_path[0] = dict_size - 2;
		for(b = 0;b < i;b++)
		{
			//reverse code & path
			dict[a].huff_code[i - b - 1] = code[b];
			dict[a].huff_path[i - b] = path[b] - dict_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}
void learnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for(a = 0; a < dict_hash_size; a++) dict_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if(fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	dict_size = 0;
	addWordToVocab((char *)"</s>");

	while(1) {
		readWord(word, fin);

		if(feof(fin)) break;
		trained_words++;

		printf("%lldK%c", trained_words / 1000, 13);
		fflush(stdout);

		i = searchVocab(word);
		if(i == -1) {
			a = addWordToVocab(word);
			dict[a].cn = 1;
		} else dict[i].cn++;

		if(dict_size > dict_hash_size * 0.7) reduceVocab();
	}

	sortVocab();

	printf("Vocab size: %lld\n", dict_size);
	printf("Words in train file: %lld\n", trained_words);
	file_size = ftell(fin);
	fclose(fin);
}

void initNet() {
	long long a, b;
	unsigned long long next_random = 1;

	a = posix_memalign((void **)&ih_weight, 128, (long long)dict_size * hidden_size * sizeof(real));
	if(ih_weight == NULL) {printf("Memory allocation failed\n");exit(1);}

	if(hs) {
		a = posix_memalign((void **)&ho_weight_hs, 128, (long long)dict_size * hidden_size * sizeof(real));
		if(ho_weight_hs == NULL) {printf("Memory allcation failed\n");exit(1);}
		for(a = 0; a < dict_size; a++) for(b = 0; b < hidden_size; b++)
			ho_weight_hs[a * hidden_size + b] = 0;
	}

	if (negative > 0) {
		a = posix_memalign((void **)&ho_weigth_neg, 128, (long long)dict_size * hidden_size * sizeof(real));
		if(ho_weigth_neg == NULL) {printf("Memory allocation failed\n");exit(1);}
		for(a = 0; a < dict_size; a++) for(b = 0; b < hidden_size; b++)
			ho_weigth_neg[a * hidden_size + b] = 0;
	}

	for(a = 0; a < dict_size; a++) for(b = 0; b < hidden_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		/* init with  ([0.5~0.5] / hidden layer size) */
		ih_weight[a * hidden_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / hidden_size;
	}
	
	assignHuffTree();
}

void *trainModelThread(void *id) {
	long long a, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH];
	long long ih_index, ho_index, c, target, label, local_iter = epochs;
	unsigned long long next_random = (long long)id;
	real out_unit, gradient;
	
	/*
	 * hidden_layer is only used by cbow, because we deal hidden layer as
	 * projection layer
	 */
	real *hidden_layer = (real *)calloc(hidden_size, sizeof(real));
	real *eh = (real *)calloc(hidden_size, sizeof(real));

	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	while(1) {
		if(word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;

			printf("%cAlpha: %f Progress %.2f%%", 13, alpha, word_count_actual / (real)(epochs * trained_words + 1) * 100);
			fflush(stdout);

			alpha = starting_alpha * (1 - word_count_actual / (real)(epochs * trained_words + 1));
			if(alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
		/* when previous loop has finished a sentence, bring new sentence */
		if(sentence_length == 0) {
			while(1) {
				word = readWordIndex(fi);
				if(feof(fi)) break;
				if(word == -1) continue;
				word_count++;

				/* if word is </s> then break */
				if(word == 0) break;

				if(sample > 0) {
					/* Subsampling of Frequent Words
					 * =================================
				     * This code randomly discards training words, but is designed to 
					 * keep the relative frequencies the same. That is, less frequent
					 * words will be discarded less often.
					 *
					 * We first calculate the probability that we want to *keep* the word;
					 * this is the value 'ran'. Then, to decide whether to keep the word,
					 * we generate a random fraction (0.0 - 1.0), and if 'ran' is smaller
					 * than this number, we discard the word. This means that the smaller 
					 * 'ran' is, the more likely it is that we'll discard this word. 
					 *
					 * The quantity (vocab[word].cn / train_words) is the fraction of all                                                    * the training words which are 'word'. Let's represent this fraction
					 * by x.
					 *
					 * Using the default 'sample' value of 0.001, the equation for ran is:
					 * ran = (sqrt(x / 0.001) + 1) * (0.001 / x)             
					 *
					 * You can plot this function to see it's behavior; it has a curved 
					 * L shape.
			 		 * 
					 * Here are some interesting points in this function (again this is
					 * using the default sample value of 0.001).
					 *   - ran = 1 (100% chance of being kept) when x <= 0.0026.
					 *
					 *   - That is, any word which is 0.0026 of the words *or fewer* 
					 * will be kept 100% of the time. Only words which represent 
					 * more than 0.26% of the total words will be subsampled.
					 *   - ran = 0.5 (50% chance of being kept) when x = 0.00746. 
				     *   - ran = 0.033 (3.3% chance of being kept) when x = 1.
					 *   - That is, if a word represented 100% of the training set
					 * (which of course would never happen), it would only be
					 *         kept 3.3% of the time.
					 *
					 * NOTE: Seems like it would be more efficient to pre-calculate this 
					 *       probability for each word and store it in the vocab table...
					 *
					 * Words that are discarded by subsampling aren't added to our training
				     * 'sentence'. This means the discarded word is neither used as an 
					 * input word or a context word for other inputs.
					 */
					next_random = next_random * (unsigned long long)25214903917 + 11;

					if(prob < (next_random & 0xFFFF) / (real)65536) continue;
				}
				sen[sentence_length] = word;
				sentence_length++;

				if(sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			sentence_position = 0;
		}
		if(feof(fi) || (word_count > trained_words / num_threads)) {
			word_count_actual += word_count - last_word_count;
			local_iter--;
			if(local_iter == 0) break;
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			continue;
		}

		word = sen[sentence_position];
		if(word == -1) continue;

		/* layer init */
		for(c = 0; c < hidden_size; c++) hidden_layer[c] = 0;
		for(c = 0; c < hidden_size; c++) eh[c] = 0;

		if(cbow) {
			/* input -> hidden */
			cw = 0;
			/* iterate sentence at the aspect of window size */
			for(a = 0; a < window * 2 + 1; a++) if (a != window) {
				c = sentence_position - window + a;
				if(c < 0) continue;
				if(c >= sentence_length) continue;
				last_word = sen[c];
				if(last_word == -1) continue;
				/* if word is valid then propagate to hidden */
				for(c = 0; c < hidden_size; c++) hidden_layer[c] += ih_weight[c + last_word * hidden_size];
				cw++;
			}
			if(cw) {
				for(c = 0; c < hidden_size; c++) hidden_layer[c] /= cw;
				if (hs) for(d = 0; d < dict[word].huff_length; d++) {
					out_unit = 0;
					ho_index = dict[word].huff_path[d] * hidden_size;

					for(c = 0; c < hidden_size; c++) out_unit += hidden_layer[c] * ho_weight_hs[c + ho_index];
					if(out_unit <= -MAX_SIG) continue;
					else if(out_unit >= MAX_SIG) continue;
					else out_unit = sig_table[(int)((out_unit + MAX_SIG) * (SIG_TABLE_SIZE / MAX_SIG / 2))];

					gradient = (1 - dict[word].huff_code[d] - out_unit) * alpha;

					/* precalculate eh/c from equation(23, 54) */
					for(c = 0; c < hidden_size; c++) eh[c] += gradient * ho_weight_hs[c + ho_index];

					/*
					 * propagate error to ho_weight
					 * equation(51)
					 */
					for(c = 0; c < hidden_size; c++) ho_weight_hs[c + ho_index] += gradient * hidden_layer[c];
				}
				if(negative > 0) for(d = 0; d < negative + 1; d++) {
					/*
					 * target is index of the word
					 * label is 1 when it's goal word, otherwise 0
					 */
					if(d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = uni_table[(next_random >> 16) % uni_table_size];
						if(target == 0) target = next_random % (dict_size - 1) + 1;
						if(target == word) continue;
						label = 0;
					}
					ho_index = target * hidden_size;
					out_unit = 0;
					for(c = 0; c < hidden_size; c++) out_unit += hidden_layer[c] * ho_weigth_neg[c + ho_index];
					if(out_unit > MAX_SIG) gradient = (label - 1) * alpha;
					else if(out_unit < -MAX_SIG) gradient = (label - 0) * alpha;
					else gradient = (label - sig_table[(int)((out_unit + MAX_SIG) * (SIG_TABLE_SIZE / MAX_SIG / 2))]) * alpha;
					/*
					 * precalculate eh/c from equation(32, 61)
					 * (label - sig(x) * ho_weight) := EH
					 */
					for(c = 0; c < hidden_size; c++) eh[c] += gradient * ho_weigth_neg[c + ho_index];

					/*
					 * propagate error to ho_layer
					 * equation (59)
					 * v`new = v`old - (sig(x) - label) * alpha
					 */
					for(c = 0; c < hidden_size; c++) ho_weigth_neg[c + ho_index] += gradient * hidden_layer[c];
				}
				//hidden -> in
				for(a = 0; a < window * 2 + 1; a++) if(a != window) {
					c = sentence_position - window + a;
					if(c < 0) continue;
					if(c >= sentence_length) continue;
					last_word = sen[c];
					if(last_word == -1) continue;
					for(c = 0; c < hidden_size; c++) ih_weight[c + last_word * hidden_size] += eh[c];
				}
			}
		}
		else {
			for(a = 0; a < window * 2 + 1; a++) if (a != window) {
				c = sentence_position - window + a;

				if(c < 0) continue;
				if(c >= sentence_length) continue;

				last_word = sen[c];
				if(last_word == -1) continue;

				ih_index = last_word * hidden_size;
				for(c = 0; c < hidden_size; c++) eh[c] = 0;

				if(hs) for(d = 0; d < dict[word].huff_length; d++) {
					out_unit = 0;
					ho_index = dict[word].huff_path[d] * hidden_size;

					for(c = 0; c < hidden_size; c++) out_unit += ih_weight[c + ih_index] * ho_weight_hs[c + ho_index];
					if(out_unit <= -MAX_SIG) continue;
					else if(out_unit >= MAX_SIG) continue;
					else out_unit = sig_table[(int)((out_unit + MAX_SIG) * (SIG_TABLE_SIZE / MAX_SIG / 2))];

					gradient = (1 - dict[word].huff_code[d] - out_unit) * alpha;

					for(c = 0; c < hidden_size; c++) eh[c] += gradient * ho_weight_hs[c + ho_index];
					for(c = 0; c < hidden_size; c++) ho_weight_hs[c + ho_index] += gradient * ih_weight[c + ih_index];
				}

				if(negative > 0) for(d = 0; d < negative + 1; d++) {
					if(d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = uni_table[(next_random >> 16) % uni_table_size];

						if(target == 0) target = next_random % (dict_size - 1) + 1;
						if(target == word) continue;

						label = 0;
					}
					ho_index = target * hidden_size;
					out_unit = 0;

					for(c = 0; c < hidden_size; c++) out_unit += ih_weight[c + ih_index] * ho_weigth_neg[c + ho_index];

					if(out_unit > MAX_SIG) gradient = (label - 1) * alpha;
					else if(out_unit < - MAX_SIG) gradient = (label - 0) * alpha;
					else gradient = (label - sig_table[(int)((out_unit + MAX_SIG) * (SIG_TABLE_SIZE / MAX_SIG / 2))]) * alpha;

					for(c = 0; c < hidden_size; c++) eh[c] += gradient * ho_weigth_neg[c + ho_index];
					for(c = 0; c < hidden_size; c++) ho_weigth_neg[c + ho_index] += gradient * ih_weight[c + ih_index];
				}
				for(c = 0; c< hidden_size; c++) ih_weight[c + ih_index] += eh[c];
			}
		}
		sentence_position++;
		if(sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(hidden_layer);
	free(eh);
	pthread_exit(NULL);
}

void trainModel() {
	long long a, b;
	FILE *fo;

	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;

	learnVocabFromTrainFile();
	if(output_file[0] == 0) return;
	initNet();

	if(negative > 0) initUnigramTable();

	for(a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, trainModelThread, (void *)a);

	for(a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	fo = fopen(output_file, "wb");
	fprintf(fo, "%lld %lld\n", dict_size, hidden_size);
	for(a = 0; a < dict_size; a++) {
		fprintf(fo, "%s ", dict[a].word);
		for(b = 0; b < hidden_size; b++) fwrite(&ih_weight[a * hidden_size + b], sizeof(real), 1, fo);
		fprintf(fo, "\n");
	}

	fclose(fo);
}

int argPos(char *str, int argc, char **argv) {
	int a;
	for(a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if(a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if(argc == 1) {
		printf("give appropriate option\n");
		return 0;
	}
	output_file[0] = 0;

	if((i = argPos((char *)"-size", argc, argv)) > 0) hidden_size = atoi(argv[i+1]);
	if((i = argPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i+1]);
	if((i = argPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i+1]);
	if(cbow) alpha = 0.05;
	if((i = argPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if((i = argPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if((i = argPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if((i = argPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if((i = argPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if((i = argPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if((i = argPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if((i = argPos((char *)"-iter", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
	if((i = argPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

	dict = (struct vocab_word *)calloc(dict_max_size, sizeof(struct vocab_word));
	dict_hash = (int *)calloc(dict_hash_size, sizeof(int));
	sig_table = (real *)malloc((SIG_TABLE_SIZE + 1) * sizeof(real));
	for(i = 0; i < SIG_TABLE_SIZE; i++) {
		sig_table[i] = exp((i / (real)SIG_TABLE_SIZE * 2 - 1) * MAX_SIG);
		sig_table[i] = sig_table[i]/ (sig_table[i] + 1);
	}
	trainModel();
	return 0;
}
