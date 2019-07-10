import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def main():
    cnt = 0
    output = open("./predicts_sigmoid_no_label_imgs.txt", "w")
    for line in open("./predicts_no_label_imgs.txt"):
        url, score = line.strip().split(",")
        new_score = sigmoid(float(score))
        output.write(url+","+str(new_score)+"\n")
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    print("convert finish")


if __name__ == "__main__":
    main()
