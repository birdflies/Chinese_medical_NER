from sklearn.metrics import classification_report


def main():
    targets = ['B', 'I', 'E', 'O', 'S']

    result_file = 'output/crf/result.txt'
    y_test = []
    y_pred = []
    with open(result_file, 'r', encoding='UTF-8') as fr:

        line = fr.readline()
        while line:
            elements = line.strip().split('\t')
            if len(elements) == 3:
                y_test.append(elements[1])
                y_pred.append(elements[2])
            else:
                print(line)
            line = fr.readline()

    report = classification_report(y_test, y_pred, target_names=targets)

    eval_file = 'output/crf/eval.txt'

    with open(eval_file, 'w+', encoding='UTF-8') as fw:
        fw.write('Classification report: \n')
        fw.write(report)


if __name__ == '__main__':
    main()
