from argparse import ArgumentParser







def main(args):
    answers = {}
    with open(args.input) as F:
        lines = F.readlines()[1:]
        for line in lines:
            name, tag = line.split(',')
            answers[name] = tag[:1]

    predictions = {}
    with open(args.output) as F:
        lines = F.readlines()
        for line in lines:
            name, tag = line.split(',')
            predictions[name] = tag[:1]

    

    count_ans = {'A':0, 'B':0, 'C':0}
    count_pred = {'A':0, 'B':0, 'C':0}
    acc = {'A':0, 'B':0, 'C':0}

    for key in answers.keys():
        ### count recall A
        count_ans[answers[key]] += 1
        count_pred[predictions[key]] += 1

        if answers[key] == predictions[key]:
            acc[answers[key]] += 1

    


    print('ans: ', count_ans)
    print('prediction: ', count_pred)
    print(acc)
    print('recallA:', acc['A']/count_ans['A'])
    print('recallB:', acc['B']/count_ans['B'])
    print('recallC:', acc['C']/count_ans['C'])
    
    
    
    
    weight = {'A':0, 'B':0, 'C':0}
    total_num = count_ans['A'] + count_ans['B'] + count_ans['C']
    for k in ['A', 'B', 'C']:
        weight[k] = count_ans[k] / total_num
    print(weight)

    WAR = 0.0
    for k in ['A', 'B', 'C']:
        ### weight * recall
        WAR += weight[k] * (acc[k]/count_ans[k])

    print(WAR)
    

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="output file")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    main(args)










