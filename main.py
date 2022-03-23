import tkinter

import gettingdata
import numpy as np
from scipy.special import expit, softmax
from scipy.stats import entropy
from PIL import Image
import random
from math import e
import os
import pickle
from tkinter import Tk, Canvas, Label

class neuronet:

    def __init__(self, in_nodes, hid_nodes, out_nodes, l_rate):
        self.in_nodes = in_nodes
        self.hid_nodes = hid_nodes
        self.out_nodes = out_nodes
        self.l_rate = l_rate
        self.w0 = np.random.rand(self.hid_nodes, self.in_nodes)
        self.w1 = np.random.rand(self.out_nodes, self.hid_nodes)
        self.w0 = self.w0 - 0.5
        self.w1 = self.w1 - 0.5

    def activation1(self, x):
        return expit(x)

    def dactivation1(self, x):
        return x * (1 - x)

    def activation2(self, x):
        return softmax(x)

    def prediction(self, inputs_list):
        inputs = np.reshape(inputs_list, (len(inputs_list), 1))
        hid_outputs = np.matmul(self.w0, inputs)
        hid_outputs = self.activation1(hid_outputs)
        out_outputs = np.matmul(self.w1, hid_outputs)
        out_outputs = self.activation2(out_outputs)
        return out_outputs

    def L(self, p, t):
        return entropy(p) + entropy(p, t)

    def dLdy(self, p, t):
        return p - t

    def dLdW1(self, p, t, x1):
        return (p - t) @ x1.transpose()

    def dLdW0(self, p, t, x1, x0):
        corr1 = self.dactivation1(x1)
        wT = self.w1.transpose()
        ans = np.matmul(wT, p - t)
        ans = ans * corr1
        ans = np.matmul(ans, x0.transpose())
        return ans

    def train(self, inputs_list, targets_list):
        inputs = np.reshape(inputs_list, (len(inputs_list), 1))
        targets = np.reshape(targets_list, (len(targets_list), 1))
        hid_outputs = np.matmul(self.w0, inputs)
        hid_outputs = self.activation1(hid_outputs)
        out_outputs = np.matmul(self.w1, hid_outputs)
        out_outputs = self.activation2(out_outputs)

        self.w0 -= self.l_rate * self.dLdW0(out_outputs, targets, hid_outputs, inputs)
        self.w1 -= self.l_rate * self.dLdW1(out_outputs, targets, hid_outputs)

def scale_input(data):
  data += 1
  data /= 257
  return data

train_data = np.array
test_data = np.array
sm = 0
add = 0

def make_data():
    arr = os.listdir("NeuralNetImages")
    global train_data
    global add
    global sm
    global test_data
    for x in arr:
        add = add + 1
        s1 = 'NeuralNetImages/' + x
        if x[0] == "t" or x[0] == "c" or x[0] == "z":
            img = Image.open(s1)
            img = img.resize((60, 60))
            adarr = np.asfarray(img.convert("L")).flatten()
            if x[0] == "t":
                adarr = np.append(0, adarr)
            if x[0] == "c":
                adarr = np.append(1, adarr)
            if x[0] == "z":
                adarr = np.append(2, adarr)
            if add == 1:
                train_data = adarr
            else:
                train_data = np.append(train_data, adarr)
            fillclr = adarr[1]
            sm = sm + 1
            num = random.randint(2, 67)
            for i in range(num):
                img = Image.open(s1)
                img = img.resize((60, 60))
                img = img.rotate(i, fillcolor=(int(fillclr), int(fillclr), int(fillclr)))
                adarr = np.asfarray(img.convert("L")).flatten()
                if x[0] == "t":
                    adarr = np.append(0, adarr)
                if x[0] == "c":
                    adarr = np.append(1, adarr)
                if x[0] == "z":
                    adarr = np.append(2, adarr)
                train_data = np.append(train_data, adarr)
                img = Image.open(s1)
                img = img.resize((60, 60))
                img = img.rotate(-i, fillcolor=(int(fillclr), int(fillclr), int(fillclr)))
                adarr = np.asfarray(img.convert("L")).flatten()
                if x[0] == "t":
                    adarr = np.append(0, adarr)
                if x[0] == "c":
                    adarr = np.append(1, adarr)
                if x[0] == "z":
                    adarr = np.append(2, adarr)
                train_data = np.append(train_data, adarr)
            sm = sm + 2 * num
            rot = random.randint(0, 80)
            img = Image.open(s1)
            img = img.resize((60, 60))
            img = img.rotate(rot, fillcolor=(int(fillclr), int(fillclr), int(fillclr)))
            adarr = np.asfarray(img.convert("L")).flatten()
            if x[0] == "t":
                adarr = np.append(0, adarr)
            if x[0] == "c":
                adarr = np.append(1, adarr)
            if x[0] == "z":
                adarr = np.append(2, adarr)
            if add == 1:
                test_data = adarr
            else:
                test_data = np.append(test_data, adarr)
    print(train_data.size)
    train_data = np.reshape(train_data, (sm, 3601))
    test_data = np.reshape(test_data, (20, 3601))
    print(train_data.size)
    for i in range(sm):
        train_data[i][1:] = scale_input(train_data[i][1:])
        if i % 10 == 0:
            print(train_data[i])
def startup():
    print(" STRATING UP")

with open("DUMP", 'rb') as readfile:
    data_new = pickle.load(readfile)


def fuckyou():
    input_nodes = 3600
    hidden_nodes = 400
    output_nodes = 3
    learning_rate = 0.0024
    epochs = 300
    net = neuronet(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(train_data.size)
    iter = 0
    for epoch in range(epochs):
        iter = 0
        for record in train_data:
            inputs = record[1:]
            targets = [0] * output_nodes
            targets[int(record[0])] = 1
            net.train(inputs, targets)
            iter = iter + 1
        log_train = []
        log_test = []
        log_ans = []
        for record in train_data:
            inputs = record[1:]
            answer = np.argmax(net.prediction(inputs))
            log_ans.append(answer)
            log_train.append(int(record[0]) == answer)
        j = 0
        np.random.shuffle(train_data)
        for x in log_train:
            j = j + x
        print(j)
        print(len(log_train))
        j = j / len(log_train)
        print(j)
        for record in test_data:
            inputs = record[1:]
            answer = np.argmax(net.prediction(inputs))
            log_ans.append(answer)
            log_test.append(int(record[0]) == answer)
        for x in log_test:
            j = j + x
        print(j)
        print(len(log_test))
        j = j / len(log_test)
        print(j, " HERE ENDS ")
    with open("DUMP", 'wb') as pickle_file:
        pickle.dump(net, pickle_file)

def draw(event):
    if event.state:
        canvas.create_oval((event.x - 2, event.y - 2),
                           (event.x + 2, event.y + 2), fill="black")
def clear(event):
    canvas.delete("all")

def getImg(fileName):
    canvas.postscript(file = fileName + '.eps')
    img = Image.open(fileName + '.eps')
    return img
def makePrediction(event):
    img = getImg("TEMPORARY")
    img = img.resize((60, 60))
    adarr = np.asfarray(img.convert("L")).flatten()


root = Tk()

canvas = tkinter.Canvas(root, width=400, height=400)
canvas.grid(row=0, column=0)
canvas.bind("<Motion>", draw)
canvas.bind("<Button-3>", clear)
canvas.bind("<Button-2>", makePrediction)
root.mainloop()

def main():
    startup()

if __name__ == "__main__":
    main()