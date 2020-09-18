# sequence-models

### Description:   
Implements three common sequential models in pure **numpy** and optionally samples from them on a character-level.  

The models include: **Recurrent Neural Network(RNN)**, **Long Short Term Memory(LSTM)**, and **Gated Recurrent Unit(GRU)**

### Usage:
```
python sequence-models.py -h
usage: sequence-models.py [-h] --m M --f F [--i I] [--h H] [--l L] [--p P]

sequence-models

optional arguments:
  -h, --help  show this help message and exit
  --m M       Model; i.e rnn,lstm,gru
  --f F       Filename; i.e dinos.txt
  --i I       iteration number; i.e 25000
  --h H       hidden nodes; i.e 50
  --l L       learning rate; i.e 0.01
  --p P       sample every 2k iterations; i.e True
  ```

### Data:
There is some sample data in the `./data` directory but you can always add your own. The code will automatically look
in this directory so only pass in the filename as a parameter.

### References:
1. https://www.coursera.org/learn/nlp-sequence-models/home/week/1 
2. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
3. http://karpathy.github.io/2015/05/21/rnn-effectiveness/
