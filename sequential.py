import numpy as np
from sequtils import *

# Disclaimer: The code is adapted from Coursera's sequence models course

# THIS FILE HOLDS CLASS DEFINITIONS FOR RNN, LSTM, and GRU SEQUENTIAL MODELS #
# @TODO : ELIMINATE REDUNDENT CODE (DRY), POTENTIAL FOR SUPERCLASS STRUCTURE

############# RNN CLASS BEGIN #############

class RNN:
    
    def __init__(self):
        np.random.seed(0)

    def initialize_parameters(self,n_a, n_x, n_y):

        Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
        Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
        Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
        b = np.zeros((n_a, 1)) # hidden bias
        by = np.zeros((n_y, 1)) # output bias
    
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
        return parameters


    def rnn_step_forward(self, parameters, a_prev, x):

        Waa = parameters['Waa']
        Wax = parameters['Wax']
        Wya = parameters['Wya']
        by  = parameters['by']
        b   = parameters['b']
    
        a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        p_t = softmax(np.dot(Wya, a_next) + by) 
    
        return a_next, p_t
    
    
    def rnn_forward(self, X, Y, a0, parameters):
        
        x, a, y_hat = {}, {}, {}
        loss = 0
        a[-1] = np.copy(a0)
    
        for t in range(len(X)):
        
            # 1 hot encoding
            x[t] = np.zeros((self.vocab_size,1))
            if(X[t] != None):
                x[t][X[t]] = 1
            
            #forward prop
            a[t], y_hat[t] = self.rnn_step_forward(parameters, a[t-1], x[t])
            
            loss -= np.log(y_hat[t][Y[t],0])
         
        cache = (y_hat, a, x)
        
        return loss, cache
    
    def rnn_step_backward(self, dy, gradients, parameters, x, a, a_prev):
    
        gradients['dWya'] += np.dot(dy, a.T)
        gradients['dby'] += dy
        da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 
        daraw = (1 - a * a) * da 
        gradients['db'] += daraw
        gradients['dWax'] += np.dot(daraw, x.T)
        gradients['dWaa'] += np.dot(daraw, a_prev.T)
        gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
        
        return gradients


    def rnn_backward(self, X, Y, parameters, cache):
        
        gradients = {}
    
        # Retrieve from cache and parameters
        (y_hat, a, x) = cache
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
        # Gradient initializations
        gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
        gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
        gradients['da_next'] = np.zeros_like(a[0])
    
        # BPTT
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t]) #softmax of y_hat at timestep t
            dy[Y[t]] -= 1 # derivative of cross entropy with respect to softmax -- dL/dy
            gradients = self.rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    
        return gradients, a

    
    def update_parameters(self,parameters, gradients, lr):

        parameters['Wax'] += -lr * gradients['dWax']
        parameters['Waa'] += -lr * gradients['dWaa']
        parameters['Wya'] += -lr * gradients['dWya']
        parameters['b']  += -lr * gradients['db']
        parameters['by']  += -lr * gradients['dby']
        
        return parameters
    
    def optimize(self, X, Y, a_prev, parameters, learning_rate):
    
        loss, cache = self.rnn_forward(X, Y, a_prev, parameters)
        gradients, a = self.rnn_backward(X, Y, parameters, cache)
        parameters = self.update_parameters(parameters, gradients, learning_rate)
    
        return loss, gradients, a[len(X)-1]
    
    def train(self, fileName, num_iterations = 10000, n_a = 50, learning_rate=0.01, predict=True):
        
        seed = 0
        np.random.seed(seed)

        X, Y, vocab_size, char_to_ix, ix_to_char = gen_data(fileName);

        n_x, n_y = vocab_size, vocab_size
        self.vocab_size = vocab_size

        parameters = self.initialize_parameters(n_a, n_x, n_y)
        
        loss_arr = []
        
        a_prev = np.zeros((n_a, 1))
        loss = -1
    
        for j in range(num_iterations):
        
            index = j % len(X)
            #X = [None] + [char_to_ix[ch] for ch in training_set[index]] 
            #Y = X[1:] + [char_to_ix["\n"]]
        
            curr_loss, gradients, a_prev = self.optimize(X[index], Y[index], a_prev, parameters, learning_rate)

            if(loss == -1):
                loss = curr_loss
            else:
                loss = loss * 0.999 + curr_loss * 0.001

            loss_arr.append(loss)

            if j % 2000 == 0:
                
                print('Iteration: %d, Loss: %f' % (j, loss_arr[j]) + '\n')
                #self.utils.save_weights(parameters, "rnn1",j)
                if(predict): 
                    for name in range(10):
                        sampled_indices = self.sample(parameters, char_to_ix, seed, vocab_size, n_a)
                        print_sample(sampled_indices, ix_to_char)
                        seed += 1
      
                    print('\n')
                    
            
        return parameters, loss_arr
    
    def clip(self, gradients, maxValue):
    
        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient,-maxValue,maxValue,gradient)

        gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    
        return gradients
    
    
    def sample(self, parameters, char_to_ix, seed, vocab_size, n_a):
    
        x = np.zeros((vocab_size,1))
        a_prev = np.zeros((n_a,1))
        
        indices = []
        idx = -1 
        counter = 0
        newline_character = char_to_ix['\n']
    
        while (idx != newline_character and counter != 50):

            #forward prop, get y output
            a_prev, y_t = self.rnn_step_forward(parameters, a_prev, x)
            
            np.random.seed(seed) 
    
            #sample from distribution
            idx = np.random.choice(list(range(vocab_size)), p=y_t.ravel())
            indices.append(idx)
 
            #1 hot encode inputs
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
        
            seed += 1
            counter +=1

        if (counter == 50):
            indices.append(char_to_ix['\n'])
    
        return indices

############# RNN CLASS END #############


############# LSTM CLASS BEGIN #############

class LSTM:
    
    def __init__(self):
        np.random.seed(0)
    
    def initialize_parameters(self, n_a, n_x, n_y):

        np.random.seed(1)
        Wf = np.random.randn(n_a, n_a + n_x)*0.01 # input to hidden
        Wi = np.random.randn(n_a, n_a + n_x)*0.01 # hidden to hidden
        Wc = np.random.randn(n_a, n_a + n_x)*0.01 # hidden to hidden
        Wo = np.random.randn(n_a, n_a + n_x)*0.01 # hidden to hidden
        Wy = np.random.randn(n_y, n_a)*0.01 # hidden to output
        bf = np.zeros((n_a, 1)) # hidden bias
        bi = np.zeros((n_a, 1))
        bc = np.zeros((n_a, 1))
        bo = np.zeros((n_a, 1))
        by = np.zeros((n_y, 1)) # output bias
    
        parameters = {"Wf": Wf, "Wi": Wi, "Wc": Wc, "Wo": Wo, "Wy": Wy, "bf": bf, "bi": bi, "bc": bc, "bo": bo, "by": by}
    
        return parameters
                      
    def lstm_step_forward(self, xt, a_prev, c_prev, parameters):

        Wf = parameters['Wf']
        Wi = parameters['Wi']
        Wc = parameters['Wc']
        Wo = parameters['Wo']
        Wy = parameters['Wy']
        bf = parameters['bf']
        bi = parameters['bi']
        bc = parameters['bc']
        bo = parameters['bo']
        by = parameters['by']
        
        n_x = xt.shape[0]
        n_a, m = a_prev.shape
        
        # stacking [a_prev, xt]
        a_prev_xt = np.zeros((n_a + n_x, m))
        a_prev_xt[: n_a, :] = a_prev
        a_prev_xt[n_a :, :] = xt

        ft = sigmoid(np.dot(Wf,a_prev_xt) + bf) #forget gate
        it = sigmoid(np.dot(Wi,a_prev_xt) + bi) #update gate
        cct = np.tanh(np.dot(Wc,a_prev_xt) + bc) #update state
        c_next = ft * c_prev + it * cct #external state
        ot = sigmoid(np.dot(Wo,a_prev_xt) + bo) #output gate 
        a_next = ot * np.tanh(c_next) #internal state
    
        yt_pred = softmax(np.dot(Wy,a_next) + by)
        
        return a_next, c_next, yt_pred, ft, it, cct, ot
    
    
    def lstm_forward(self, X, Y, a0, c0, parameters):
        
        x, a, c, y_hat = {}, {}, {}, {}
        ft, it, cct, ot = {}, {}, {}, {}
    
        a[-1] = np.copy(a0)
        c[-1] = np.copy(c0)
    
        loss = 0

        for t in range(len(X)):
            x[t] = np.zeros((self.vocab_size,1)) 
            
            if(X[t] != None):
                x[t][X[t]] = 1
                
            a[t], c[t], y_hat[t], ft[t], it[t], cct[t], ot[t] = self.lstm_step_forward(x[t], a[t-1], c[t-1], parameters)
        
            loss -= np.log(y_hat[t][Y[t],0])
            
        cache = (x, y_hat, a, c, ft, it, cct, ot)

        return loss, cache


    def lstm_cell_backward(self, dy, gradients, parameters, cache_t):
 
        (x,y_hat,a,a_prev,c,c_prev,ft,it,cct,ot) = cache_t
        
        n_x, m = x.shape
        n_a, m = a.shape
        
        gradients['dWy'] += np.dot(dy, a.T)
        gradients['dby'] += dy
        da_next = np.dot(parameters['Wy'].T, dy) + gradients['da_next']
        dc_next = gradients['dc_next']
    
        #gate derivatives
        dot = da_next * np.tanh(c) * ot * (1 - ot)
        dcct = (da_next * ot * (1 - np.tanh(c) ** 2) + dc_next) * it * (1 - cct ** 2)
        dit = (da_next * ot * (1 - np.tanh(c) ** 2) + dc_next) * cct * (1 - it) * it
        dft = (da_next * ot * (1 - np.tanh(c) ** 2) + dc_next) * c_prev * ft * (1 - ft)
    
        #weight derivatives
        gradients['dWf'] += np.dot(dft, np.hstack([a_prev.T, x.T]))
        gradients['dWi'] += np.dot(dit, np.hstack([a_prev.T, x.T]))
        gradients['dWc'] += np.dot(dcct, np.hstack([a_prev.T, x.T]))
        gradients['dWo'] += np.dot(dot, np.hstack([a_prev.T, x.T]))
        gradients['dbf'] += np.sum(dft, axis=1, keepdims=True)
        gradients['dbi'] += np.sum(dit, axis=1, keepdims=True)
        gradients['dbc'] += np.sum(dcct, axis=1, keepdims=True)
        gradients['dbo'] += np.sum(dot, axis=1, keepdims=True)

        
        gradients['da_next'] = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wc'][:, :n_a].T, dcct) 
        + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wo'][:, :n_a].T, dot)
        
        gradients['dc_next'] = (da_next * ot * (1 - np.tanh(c) ** 2) + dc_next) * ft
        
        gradients['dxt'] = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wc'][:, n_a:].T, dcct) 
        + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    
        return gradients
    
        

    def lstm_backward(self, X, Y, parameters, cache):
    
        x, y_hat, a, c, ft, it, cct, ot = cache
    
        Wf = parameters['Wf']
        Wi = parameters['Wi']
        Wc = parameters['Wc']
        Wo = parameters['Wo']
        Wy = parameters['Wy']
        bf = parameters['bf']
        bi = parameters['bi']
        bc = parameters['bc']
        bo = parameters['bo']
        by = parameters['by']
    
        gradients = {}
        
        gradients['dx'] = np.zeros_like(x)
        
        gradients['da_next'] = np.zeros_like(a[0]) 
        gradients['dc_next'] = np.zeros_like(c[0])
    
        gradients['dWf'] = np.zeros_like(Wf)
        gradients['dWi'] = np.zeros_like(Wi) 
        gradients['dWc'] = np.zeros_like(Wc)
        gradients['dWo'] = np.zeros_like(Wo)
        gradients['dWy'] = np.zeros_like(Wy)
        gradients['dbf'] = np.zeros_like(bf)
        gradients['dbi'] = np.zeros_like(bi)
        gradients['dbc'] = np.zeros_like(bc)
        gradients['dbo'] = np.zeros_like(bo)
        gradients['dby'] = np.zeros_like(by)
    
        #BPTT
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            cache_t = (x[t],y_hat[t],a[t],a[t-1],c[t],c[t-1],ft[t],it[t],cct[t],ot[t])
            gradients = self.lstm_cell_backward(dy, gradients, parameters, cache_t)
        
        return gradients, a, c

    
    def update_parameters(self, parameters, gradients, lr):
        
        parameters['Wf'] += -lr * gradients['dWf']
        parameters['Wi'] += -lr * gradients['dWi']
        parameters['Wc'] += -lr * gradients['dWc']
        parameters['Wo'] += -lr * gradients['dWo']
        parameters['Wy'] += -lr * gradients['dWy']
        parameters['bf'] += -lr * gradients['dbf']
        parameters['bi'] += -lr * gradients['dbi']
        parameters['bc'] += -lr * gradients['dbc']
        parameters['bo'] += -lr * gradients['dbo']
        parameters['by'] += -lr * gradients['dby']
        
        return parameters
        
    def optimize(self, X, Y, a_prev, c_prev, parameters, learning_rate):
    
        loss, cache = self.lstm_forward(X, Y, a_prev, c_prev, parameters)
        gradients, a, c = self.lstm_backward(X, Y, parameters, cache)
        parameters = self.update_parameters(parameters, gradients, learning_rate)
    
        return loss, gradients, a[len(X)-1], c[len(X)-1]
    
    def train(self, fileName, num_iterations = 10000, n_a = 50, learning_rate = 0.01, predict=True):

        seed = 0
        np.random.seed(seed)

        X, Y, vocab_size, char_to_ix, ix_to_char = gen_data(fileName);

        n_x, n_y = vocab_size, vocab_size
        self.vocab_size = vocab_size

        parameters = self.initialize_parameters(n_a, n_x, n_y)
        
        loss_arr = []

        # Initialize the hidden state of your GRU
        a_prev = np.zeros((n_a, 1))
        c_prev = np.zeros((n_a, 1))
        loss = -1

    
        # Optimization loop
        for j in range(num_iterations):
        
            index = j % len(X)
            #X = [None] + [char_to_ix[ch] for ch in training_set[index]] 
            #Y = X[1:] + [char_to_ix["\n"]]
        
            curr_loss, gradients, a_prev, c_prev = self.optimize(X[index], Y[index], a_prev, c_prev, parameters, learning_rate)
    
            if(loss == -1):
                loss = curr_loss
            else:
                loss = loss * 0.999 + curr_loss * 0.001

            loss_arr.append(loss)

            if j % 2000 == 0:
                print('Iteration: %d, Loss: %f' % (j, loss_arr[j]) + '\n')
                
                if(predict): 
                    for name in range(10):
                        sampled_indices = self.sample(parameters, char_to_ix, seed, vocab_size, n_a)
                        print_sample(sampled_indices, ix_to_char)
                        seed += 1
      
                    print('\n')
                
        
        return parameters, loss_arr
    
    
    def sample(self, parameters, char_to_ix, seed, vocab_size, n_a):
    
        x = np.zeros((vocab_size,1))
        a_prev = np.zeros((n_a,1))
        c_prev = np.zeros((n_a,1))
        
        indices = []
        idx = -1 
        counter = 0
        newline_character = char_to_ix['\n']
    
        while (idx != newline_character and counter != 50):
            
            a_prev, c_prev, y_t, ft, it, cct, ot = self.lstm_step_forward(x,a_prev,c_prev,parameters)
            
            
            np.random.seed(seed+counter) 
    
            idx = np.random.choice(list(range(vocab_size)), p=y_t.ravel())
            indices.append(idx)
 
            #1 hot encode inputs
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
        
            seed += 1
            counter +=1

        if (counter == 50):
            indices.append(char_to_ix['\n'])
    
        return indices

############# LSTM CLASS END #############


############# GRU CLASS BEGIN #############

class GRU:

    def __init__(self):
        np.random.seed(2)

    def initialize_parameters(self, n_c, n_x, n_y):

        Wz = np.random.randn(n_c, n_c + n_x) * 0.01  # input to hidden
        Wr = np.random.randn(n_c, n_c + n_x) * 0.01  # hidden to hidden
        Wch = np.random.randn(n_c, n_c + n_x) * 0.01  # hidden to hidden
        Wy = np.random.randn(n_y, n_c) * 0.01  # hidden to output

        bz = np.zeros((n_c, 1))  # hidden br as
        br = np.zeros((n_c, 1))
        bch = np.zeros((n_c, 1))
        by = np.zeros((n_y, 1))  # output br as

        parameters = {"Wz": Wz, "Wr": Wr, "Wch": Wch, "Wy": Wy, "bz": bz, "br": br, "bch": bch, "by": by}

        return parameters

    def gru_step_forward(self, xt, c_prev, parameters):

        Wz = parameters['Wz']
        Wr = parameters['Wr']
        Wch = parameters['Wch']
        Wy = parameters['Wy']
        bz = parameters['bz']
        br = parameters['br']
        bch = parameters['bch']
        by = parameters['by']

        '''
        print("Weights:")
        print("Wz: ", Wz.mean())
        print("Wr: ", Wr.mean())
        print("Wch: ", Wch.mean())
        print("Wy: ", Wy.mean())
        print()   
        '''

        n_x = xt.shape[0]
        n_c, m = c_prev.shape

        c_prev_xt = np.zeros((n_c + n_x, m))
        c_prev_xt[: n_c, :] = c_prev
        c_prev_xt[n_c:, :] = xt

        zt = sigmoid(np.dot(Wz, c_prev_xt) + bz)
        rt = sigmoid(np.dot(Wr, c_prev_xt) + br)

        c_prev_rt_xt = np.vstack([np.multiply(rt, c_prev_xt[: n_c, :]), xt])

        ch = np.tanh(np.dot(Wch, c_prev_rt_xt) + bch)
        c_next = np.multiply(zt, c_prev) + np.multiply( 1 - zt,  ch)

        yt_pred = softmax(np.dot(Wy, c_next) + by)

        return c_next, yt_pred, zt, rt, ch

    def gru_forward(self, X, Y, c0, parameters):

        x, c, y_hat = {}, {}, {}
        zt, rt, ch = {}, {}, {}

        c[-1] = np.copy(c0)

        loss = 0

        for t in range(len(X)):

            x[t] = np.zeros((self.vocab_size, 1))

            if (X[t] != None):
                x[t][X[t]] = 1

            c[t], y_hat[t], zt[t], rt[t], ch[t] = self.gru_step_forward(x[t], c[t - 1], parameters)

            loss -= np.log(y_hat[t][Y[t], 0])

        cache = (x, y_hat, c, zt, rt, ch)

        return loss, cache

    def gru_cell_backward(self, dy, gradients, parameters, cache_t):

        (x, y_hat, c, c_prev, zt, rt, ch) = cache_t

        n_x, m = x.shape
        n_c, m = c.shape

        gradients['dWy'] += np.dot(dy, c.T)
        gradients['dby'] += dy

        dc_next = np.dot(parameters['Wy'].T, dy) + gradients['dc_next']
        dcht = dc_next * (1 - zt) * (1 - ch ** 2)

        drc = np.dot(parameters['Wch'][:, : n_c].T, dcht)
        dr = np.multiply(drc, c_prev)
        drt = dr * rt * (1 - rt)

        dz = np.multiply(dc_next, c_prev - ch)
        dzt = dz * zt * (1 - zt)


        gradients['dWch'][:, n_c:] += np.dot(dcht, x.T)
        gradients['dWch'][:, : n_c] += np.dot(dcht, (rt * c_prev).T)
        gradients['dWr'] += np.dot(drt, np.hstack([c_prev.T,  x.T]))
        gradients['dWz'] += np.dot(dzt,  np.hstack([c_prev.T, x.T]))
        gradients['dbch'] += np.sum(dcht, axis=1, keepdims=True)
        gradients['dbz'] += np.sum(dzt, axis=1, keepdims=True)
        gradients['dbr'] += np.sum(drt, axis=1, keepdims=True)


        dh1 = np.dot(parameters['Wz'][:, : n_c].T, dzt)
        dh2 = np.dot(parameters['Wr'][:, : n_c].T, drt)
        dh3 = np.multiply(dc_next, zt)
        dh4 = np.multiply(drc, rt)

        gradients['dc_next'] = dh1 + dh2 + dh3 + dh4

        return gradients


    def gru_backward(self, X, Y, parameters, cache):

        x, y_hat, c, zt, rt, ch = cache

        Wz = parameters['Wz']
        Wr = parameters['Wr']
        Wch = parameters['Wch']
        Wy = parameters['Wy']
        bz = parameters['bz']
        br = parameters['br']
        bch = parameters['bch']
        by = parameters['by']

        gradients = {}

        gradients['dx'] = np.zeros_like(x)

        gradients['dc_next'] = np.zeros_like(c[0])

        gradients['dWz'] = np.zeros_like(Wz)
        gradients['dWr'] = np.zeros_like(Wr)
        gradients['dWch'] = np.zeros_like(Wch)
        gradients['dWy'] = np.zeros_like(Wy)
        gradients['dbz'] = np.zeros_like(bz)
        gradients['dbr'] = np.zeros_like(br)
        gradients['dbch'] = np.zeros_like(bch)
        gradients['dby'] = np.zeros_like(by)

        # backprop through time
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            cache_t = (x[t], y_hat[t], c[t], c[t - 1], zt[t], rt[t], ch[t])
            gradients = self.gru_cell_backward(dy, gradients, parameters, cache_t)

        return gradients, c

    def update_parameters(self, parameters, gradients, lr):
        '''
        print("Gradients:")
        print("dWz: ", gradients['dWz'].mean())
        print("dWr: ", gradients['dWr'].mean())
        print("dWch: ", gradients['dWch'].mean())
        print("dWy: ", gradients['dWy'].mean())
        print()
        '''

        parameters['Wz'] += -lr * gradients['dWz']
        parameters['Wr'] += -lr * gradients['dWr']
        parameters['Wch'] += -lr * gradients['dWch']
        parameters['Wy'] += -lr * gradients['dWy']
        parameters['bz'] += -lr * gradients['dbz']
        parameters['br'] += -lr * gradients['dbr']
        parameters['bch'] += -lr * gradients['dbch']
        parameters['by'] += -lr * gradients['dby']

        return parameters

    def optimize(self, X, Y, c_prev, parameters, learning_rate):

        loss, cache = self.gru_forward(X, Y, c_prev, parameters)
        gradients, c = self.gru_backward(X, Y, parameters, cache)
        parameters = self.update_parameters(parameters, gradients, learning_rate)

        return loss, gradients, c[len(X) - 1]

    def train(self, fileName, num_iterations = 10000, n_c = 50, learning_rate = 0.01, predict=True):

        seed = 0
        np.random.seed(seed)

        X, Y, vocab_size, char_to_ix, ix_to_char = gen_data(fileName);

        n_x, n_y = vocab_size, vocab_size
        self.vocab_size = vocab_size

        parameters = self.initialize_parameters(n_c, n_x, n_y)
        
        loss_arr = []

        # Initialize the hidden state of your GRU
        c_prev = np.zeros((n_c, 1))
        loss = -1

        # Optimization loop
        for j in range(num_iterations):

            index = j % len(X)
            #X = [None] + [char_to_ix[ch] for ch in training_set[index]]
            #Y = X[1:] + [char_to_ix["\n"]]

            # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, c_prev = self.optimize(X[index], Y[index], c_prev, parameters, learning_rate)

            if(loss == -1):
                loss = curr_loss
            else:
                loss = loss * 0.999 + curr_loss * 0.001

            loss_arr.append(loss)

            if j % 2000 == 0:
                print('Iteration: %d, Loss: %f' % (j, loss_arr[j]) + '\n')

                if (predict):
                    for name in range(10):
                        sampled_indices = self.sample(parameters, char_to_ix, seed, vocab_size, n_c)
                        print_sample(sampled_indices, ix_to_char)
                        seed += 1

                    print('\n')

        return parameters, loss_arr

    def sample(self, parameters, char_to_ix, seed, vocab_size, n_a):

        x = np.zeros((vocab_size, 1))
        c_prev = np.zeros((n_a, 1))

        indices = []
        idx = -1
        counter = 0
        newline_character = char_to_ix['\n']

        while (idx != newline_character and counter != 50):

            c_prev, y_t, zt, rt, ch = self.gru_step_forward(x, c_prev, parameters)

            np.random.seed(seed + counter)

            idx = np.random.choice(list(range(vocab_size)), p=y_t.ravel())
            indices.append(idx)

            # 1 hot encode inputs
            x = np.zeros((vocab_size, 1))
            x[idx] = 1

            seed += 1
            counter += 1

        if (counter == 50):
            indices.append(char_to_ix['\n'])

        return indices
    

############# GRU CLASS END #############



