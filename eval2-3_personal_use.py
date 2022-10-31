import csv
import argparse

# python3 eval2-3.py

parser = argparse.ArgumentParser(description="hw 2-1 train",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("ans", help="")
parser.add_argument("pred", help="")


args = parser.parse_args()
print(vars(args))

pred = args.pred 
ans = args.ans

def read_csv(path):
    label = []
    filename = []
    with open(f'{path}', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        for i, row in enumerate(rows):
            if i==0: 
                continue
            fname, l = row
            label.append(l)
            filename.append(fname)

    return label, filename


print(ans)
print(pred)
ans_label, ans_filename = read_csv(ans)
pred_label, pred_filename = read_csv(pred)

total = 0
correct = 0
for i, ans_fname in enumerate(ans_filename):
    assert ans_fname in pred_filename

    j = pred_filename.index(ans_fname)
    if ans_label[i] == pred_label[j]:
        correct+=1

    total+=1
print("{:.3f}".format(correct/total))

