import numpy as np
import matplotlib.pyplot as plt
from sequential import RNN, LSTM, GRU
from sequtils import *
import argparse


def main():
	parser = argparse.ArgumentParser(description='sequence-models')
	parser.add_argument('--m',required=True,help="Model; i.e rnn,lstm,gru")
	parser.add_argument('--f',required=True,help="Filename; i.e dinos.txt")
	parser.add_argument('--i',required=False,default=10000,help="iteration number; i.e 25000")
	parser.add_argument('--h',required=False,default=50,help="hidden nodes; i.e 50")
	parser.add_argument('--l',required=False,default=0.01,help="learning rate; i.e 0.01")
	parser.add_argument('--p',required=False,default=True,help="sample every 2k iterations; i.e True")

	args = parser.parse_args()

	if(args.m == "rnn"):
		model = RNN()
	elif(args.m == "gru"):
		model = GRU()
	elif(args.m == "lstm"):
		model = LSTM()
	else:
		print("Invalid model choice")
		exit()

	plt.figure()
	parameters, loss = model.train(args.f, args.i, args.h, args.l, args.p)
	plt.plot(loss)
	plt.title('Loss over Iterations')
	plt.ylabel('loss')
	plt.xlabel('iterations')
	plt.show()
	plt.figure()

if __name__ == '__main__':
	main()



