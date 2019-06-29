from sklearn.metrics import classification_report


label_test_file = 'output/crf/result.txt'
# label_pred_file = 'output/crf/label_pred.txt'
eval_file = 'output/crf/eval_crf.txt'

# label_test_file = 'output/crf/label_test.txt'
# label_pred_file = 'output/crf/label_pred.txt'
# eval_file = 'output/crf/eval_bert.txt'


def main():
    targets = ['B', 'I', 'E', 'O', 'S']
    y_test = []
    y_pred = []
    with open(label_test_file, 'r', encoding='UTF-8') as fr:

        line = fr.readline()
        while line:
            elements = line.strip().split('\t')
            if len(elements) == 3:
                y_test.append(elements[1])
                y_pred.append(elements[2])
            else:
                print(line)
            line = fr.readline()

    # with open(label_pred_file, 'r', encoding='UTF-8') as fr:
    #
    #     line = fr.readline()
    #     while line:
    #         element = line.strip()
    #         if element == '[SEP]' or element =='[CLS]':
    #             pass
    #         else:
    #             y_pred.append(element)
    #         line = fr.readline()

    print('Test: {}\nPred: {}'.format(len(y_test), len(y_pred)))

    report = classification_report(y_test, y_pred, digits=4, target_names=targets)

    with open(eval_file, 'w+', encoding='UTF-8') as fw:
        fw.write('Classification report: \n')
        fw.write(report)


if __name__ == '__main__':
    main()
