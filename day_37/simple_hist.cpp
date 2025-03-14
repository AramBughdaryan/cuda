

void histogram_squential(char *data, unsigned int length, unsigned int *histo){
    for (unsigned int i = 0; i < length; i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position/4]++;
        }
    }   
}