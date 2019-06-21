import time


class Char2Tag:
    def __init__(self, input_file, output_file, target):
        self.input_file = input_file
        self.output_file = output_file
        self.target = target

    def tagging(self, char, tag):
        self.fw.write("%s %s\n" % (char, tag))

    def char2tag(self):
        self.fw = open(self.output_file, 'w+', encoding='utf-8')

        with open(self.input_file, 'r', encoding='utf-8') as fr:
            line = fr.readline()
            title_id = None
            while line:
                elements = line.strip().split('\t')
                if len(elements) != 4:
                    print(line, len(elements))
                    continue
                if elements[0] != title_id:
                    title_id = elements[0]
                    self.tagging('。', 'O')
                    self.fw.write('\n')
                word = elements[1]
                cixing = elements[3]
                word_len = len(word)
                if cixing == self.target:
                    if word_len == 1:
                        char = word
                        tag = 'S'
                        self.tagging(char, tag)
                    else:
                        for char_index, char in enumerate(word):
                            if char_index == 0:
                                tag = 'B'
                                self.tagging(char, tag)
                            elif char_index == word_len - 1:
                                tag = 'E'
                                self.tagging(char, tag)
                            elif word_len > 2 and char_index < word_len - 1:
                                tag = 'I'
                                self.tagging(char, tag)
                else:
                    for char_index, char in enumerate(word):
                        tag = 'O'
                        self.tagging(char, tag)
                line = fr.readline()

        self.fw.close()


if __name__ == '__main__':
    mode = 'test'
    file_input = '../data/bxb/{}_records.txt'.format(mode)
    file_output = '../data/bxb/{}_bioes_char.txt'.format(mode)
    target_cixing = 'nz'

    start_time = time.time()
    print('Start tagging...')

    file = Char2Tag(file_input, file_output, target_cixing)
    file.char2tag()

    print('Finished in %f' % (time.time() - start_time))
